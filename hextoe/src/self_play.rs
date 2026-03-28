use rand::Rng;
use std::time::{Duration, Instant};

use candle_core::Device;

use crate::encode::{action_to_index, board_center, encode_state, CHANNELS, GRID};
use crate::game::{GameState, Player, Pos};
use crate::mcts::{Mcts, NaiveRollout, RolloutPolicy, MAX_GAME_MOVES};
use crate::nn::{DualNetRollout, HextoeNet};
use crate::nnue::encode_nnue;

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
    /// NNUE sparse feature indices (u16; fits since N_FEATURES < 65536).
    /// Empty only for records loaded from pre-NNUE checkpoints.
    pub nnue_feats: Vec<u16>,
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
        rollout: &P,
    ) -> Vec<GameRecord> {
        self.play_game_with_progress(mcts_iters, rng, rollout, |_, _| {})
    }

    /// Like [`Self::play_game`], but calls `on_mcts` after each MCTS search with the
    /// 1-based move index and time spent in MCTS for that position (useful for logging).
    pub fn play_game_with_progress<R, P, F>(
        &self,
        mcts_iters: u32,
        rng: &mut R,
        rollout: &P,
        mut on_mcts: F,
    ) -> Vec<GameRecord>
    where
        R: Rng,
        P: RolloutPolicy,
        F: FnMut(u32, Duration),
    {
        let mut state = GameState::new();
        // Temporary storage: (encoded_state, pi, player_to_move, nnue_feats)
        let mut steps: Vec<([f32; CHANNELS * GRID * GRID], [f32; GRID * GRID], Player, Vec<u16>)> =
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
            let nnue_feats: Vec<u16> =
                encode_nnue(&state, center).into_iter().map(|f| f as u16).collect();

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

            steps.push((state_enc, pi, current_player, nnue_feats));
            state.place(chosen_pos);
            move_count += 1;
        }

        // ----- assign outcomes -----
        // Decisive games use +1/-1. Non-terminal games (MAX_GAME_MOVES reached) use a
        // threat-based heuristic so the value head still gets a useful training signal.
        let winner = state.winner;
        steps
            .into_iter()
            .map(|(state_enc, pi, player, nnue_feats)| {
                let outcome = match winner {
                    Some(w) if w == player => 1.0,
                    Some(_) => -1.0,
                    None => state.board_heuristic(player),
                };
                GameRecord {
                    state_enc: Box::new(state_enc),
                    pi: Box::new(pi),
                    outcome,
                    nnue_feats,
                }
            })
            .collect()
    }

    /// Play one game where `naive_player` uses a greedy [`NaiveRollout`] (no MCTS — just
    /// argmax of own-run-extension priors) and the other player uses full MCTS with
    /// `rollout`. Records are collected for both sides so the trained bot sees the naive
    /// bot's positions with their outcomes.
    pub fn play_game_vs_naive<R: Rng, P: RolloutPolicy>(
        &self,
        mcts_iters: u32,
        rng: &mut R,
        rollout: &P,
        naive_player: Player,
    ) -> Vec<GameRecord> {
        let naive = NaiveRollout;
        let mut state = GameState::new();
        let mut steps: Vec<([f32; CHANNELS * GRID * GRID], [f32; GRID * GRID], Player, Vec<u16>)> =
            Vec::new();
        let mut move_count = 0u32;

        loop {
            if state.is_terminal() || move_count >= MAX_GAME_MOVES {
                break;
            }
            let state_enc = encode_state(&state);
            let center = board_center(&state);
            let current_player = state.current_player();
            let nnue_feats: Vec<u16> =
                encode_nnue(&state, center).into_iter().map(|f| f as u16).collect();

            let chosen_pos = if current_player == naive_player {
                // Naive player: pick argmax of own-run-extension priors, no MCTS.
                naive.priors_only(&state)
                    .and_then(|priors| priors.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()))
                    .map(|(pos, _)| pos)
                    .unwrap_or_else(|| state.legal_actions()[0])
            } else {
                // Trained player: full MCTS.
                let mut mcts = Mcts::new(state.clone());
                mcts.search_iters(mcts_iters, rng, rollout);
                let stats = mcts.root_children_stats();
                let total: u32 = stats.iter().map(|(_, v)| v).sum();
                if total == 0 {
                    state.legal_actions()[0]
                } else {
                    let threshold = rng.gen::<f32>() * total as f32;
                    let mut cum = 0.0f32;
                    let mut chosen = stats[0].0;
                    for &(pos, visits) in &stats {
                        cum += visits as f32;
                        if cum >= threshold { chosen = pos; break; }
                    }
                    chosen
                }
            };

            // Build pi from a one-hot on chosen_pos for the naive player,
            // or from MCTS visit counts for the trained player (already sampled above).
            let mut pi = [0.0f32; GRID * GRID];
            if let Some(idx) = action_to_index(chosen_pos, center) {
                pi[idx] = 1.0;
            }

            steps.push((state_enc, pi, current_player, nnue_feats));
            state.place(chosen_pos);
            move_count += 1;
        }

        let winner = state.winner;
        steps
            .into_iter()
            .map(|(state_enc, pi, player, nnue_feats)| {
                let outcome = match winner {
                    Some(w) if w == player => 1.0,
                    Some(_) => -1.0,
                    None => state.board_heuristic(player),
                };
                GameRecord { state_enc: Box::new(state_enc), pi: Box::new(pi), outcome, nnue_feats }
            })
            .collect()
    }

    /// One full game between two networks (MCTS + neural rollouts). `new_player` is which
    /// side uses `new_net`; the other uses `best_net`. Returns the winner, if any.
    pub fn play_match_game<R: Rng>(
        &self,
        mcts_iters: u32,
        rng: &mut R,
        new_net: &HextoeNet,
        best_net: &HextoeNet,
        new_player: Player,
        device: &Device,
    ) -> Option<Player> {
        let mut state = GameState::new();
        let mut move_count = 0u32;
        let dual = DualNetRollout {
            new_net,
            best_net,
            new_player,
            device,
        };

        loop {
            if state.is_terminal() {
                break;
            }
            if move_count >= MAX_GAME_MOVES {
                break;
            }

            let mut mcts = Mcts::new(state.clone());
            mcts.search_iters(mcts_iters, rng, &dual);

            let children_stats: Vec<(Pos, u32)> = mcts.root_children_stats();
            let total_visits: u32 = children_stats.iter().map(|(_, v)| v).sum();

            let chosen_pos = if total_visits == 0 {
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

            state.place(chosen_pos);
            move_count += 1;
        }

        // Decisive winner takes precedence; for non-terminal games, use the board
        // heuristic so tournaments can still differentiate candidates.
        match state.winner {
            w @ Some(_) => w,
            None => {
                let h = state.board_heuristic(Player::X);
                if h > 0.01 {
                    Some(Player::X)
                } else if h < -0.01 {
                    Some(Player::O)
                } else {
                    None
                }
            }
        }
    }
}

impl Default for SelfPlayCollector {
    fn default() -> Self {
        Self::new()
    }
}
