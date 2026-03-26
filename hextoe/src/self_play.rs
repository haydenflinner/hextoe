use rand::Rng;
use std::time::{Duration, Instant};

use crate::encode::{action_to_index, board_center, encode_state, CHANNELS, GRID};
use crate::game::{GameState, Player, Pos};
use crate::mcts::{Mcts, RolloutPolicy, MAX_GAME_MOVES};

// ── Data structures ───────────────────────────────────────────────────────────

/// A single training example produced during self-play.
pub struct GameRecord {
    /// Encoded board state at the time the move was chosen.
    pub state_enc: Box<[f32; CHANNELS * GRID * GRID]>,
    /// MCTS visit-count distribution over the `GRID * GRID` flat action grid.
    pub pi: Box<[f32; GRID * GRID]>,
    /// Game outcome from the perspective of the player to move at this position:
    /// +1.0 = win, -1.0 = loss, 0.0 = draw.
    pub outcome: f32,
}

// ── Replay buffer ─────────────────────────────────────────────────────────────

/// Circular buffer that stores `GameRecord`s up to a fixed capacity.
pub struct ReplayBuffer {
    records: Vec<GameRecord>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            records: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a record; if at capacity, drop the oldest entry first.
    pub fn push(&mut self, record: GameRecord) {
        if self.records.len() == self.capacity {
            self.records.remove(0);
        }
        self.records.push(record);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Sample `batch_size` records without replacement.
    /// If `batch_size > len()` the sample is taken with replacement instead.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&GameRecord> {
        if self.records.is_empty() {
            return vec![];
        }
        let n = self.records.len();
        if batch_size >= n {
            // With replacement.
            (0..batch_size)
                .map(|_| &self.records[rng.gen_range(0..n)])
                .collect()
        } else {
            // Without replacement: partial Fisher-Yates over an index array.
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..batch_size {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
            }
            indices[..batch_size]
                .iter()
                .map(|&i| &self.records[i])
                .collect()
        }
    }
}

// ── Self-play collector ───────────────────────────────────────────────────────

pub struct SelfPlayCollector;

impl SelfPlayCollector {
    pub fn new() -> Self {
        SelfPlayCollector
    }

    /// Play one complete game using pure MCTS and collect training records.
    ///
    /// For every position:
    ///   1. Run `mcts_iters` MCTS iterations.
    ///   2. Build the π vector from root children visit counts.
    ///   3. Sample the next move proportional to visit counts (τ=1).
    ///
    /// After the game terminates, set each record's `outcome` to +1/-1/0
    /// from the perspective of the player who was to move at that step.
    pub fn play_game<R: Rng, P: RolloutPolicy>(
        &self,
        mcts_iters: u32,
        rng: &mut R,
        rollout: &mut P,
    ) -> Vec<GameRecord> {
        self.play_game_with_progress(mcts_iters, rng, rollout, |_, _| {})
    }

    /// Like [`Self::play_game`], but calls `on_mcts` after each MCTS search with the
    /// 1-based move index and time spent in MCTS for that position (useful for logging).
    pub fn play_game_with_progress<R, P, F>(
        &self,
        mcts_iters: u32,
        rng: &mut R,
        rollout: &mut P,
        mut on_mcts: F,
    ) -> Vec<GameRecord>
    where
        R: Rng,
        P: RolloutPolicy,
        F: FnMut(u32, Duration),
    {
        let mut state = GameState::new();
        // Temporary storage: (encoded_state, pi, player_to_move)
        let mut steps: Vec<([f32; CHANNELS * GRID * GRID], [f32; GRID * GRID], Player)> =
            Vec::new();
        let mut move_count = 0u32;

        loop {
            if state.is_terminal() {
                break;
            }
            if move_count >= MAX_GAME_MOVES {
                break;
            }

            // ----- encode current state -----
            let state_enc = encode_state(&state);
            let center = board_center(&state);
            let current_player = state.current_player();

            // ----- run MCTS -----
            let mut mcts = Mcts::new(state.clone());
            let t_mcts = Instant::now();
            mcts.search_iters(mcts_iters, rng, rollout);
            let mcts_dt = t_mcts.elapsed();
            on_mcts(move_count + 1, mcts_dt);

            let children_stats: Vec<(Pos, u32)> = mcts.root_children_stats();

            // ----- build π -----
            let mut pi = [0.0f32; GRID * GRID];
            let total_visits: u32 = children_stats.iter().map(|(_, v)| v).sum();

            if total_visits > 0 {
                for &(pos, visits) in &children_stats {
                    if let Some(idx) = action_to_index(pos, center) {
                        pi[idx] = visits as f32 / total_visits as f32;
                    }
                }
            }

            // ----- sample move proportional to visit counts (τ=1) -----
            let chosen_pos = if total_visits == 0 {
                // Fallback: uniform over legal actions.
                let actions = state.legal_actions();
                actions[rng.gen_range(0..actions.len())]
            } else {
                let threshold = rng.gen::<f32>() * total_visits as f32;
                let mut cumulative = 0.0f32;
                let mut chosen = children_stats[0].0;
                for &(pos, visits) in &children_stats {
                    cumulative += visits as f32;
                    if cumulative >= threshold {
                        chosen = pos;
                        break;
                    }
                }
                chosen
            };

            steps.push((state_enc, pi, current_player));
            state.place(chosen_pos);
            move_count += 1;
        }

        // ----- assign outcomes -----
        let winner = state.winner;
        steps
            .into_iter()
            .map(|(state_enc, pi, player)| {
                let outcome = match winner {
                    Some(w) if w == player => 1.0,
                    Some(_) => -1.0,
                    None => 0.0,
                };
                GameRecord {
                    state_enc: Box::new(state_enc),
                    pi: Box::new(pi),
                    outcome,
                }
            })
            .collect()
    }
}

impl Default for SelfPlayCollector {
    fn default() -> Self {
        Self::new()
    }
}
