/// Specialized MCTS for Hextoe.
///
/// Key differences from the generic mcts-rs reference:
/// - Node arena uses `Vec<Node>` with index-based links (no Box/Rc).
/// - `GameState` maintains an incremental `candidates` set, so legal-action
///   enumeration never rescans the full board.
/// - Rewards are stored from a fixed root-player perspective, avoiding
///   per-level sign bookkeeping that breaks for 2-moves-per-turn games.
use crate::game::{max_run_through, opp_straight_extension_blocks, runs_per_axis, GameState, Player, Pos};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;

/// Exploration constant for UCB1 (random rollout path).
const C: f32 = std::f32::consts::SQRT_2;
/// Exploration constant for PUCT (NN policy path).
const C_PUCT: f32 = 2.0;

/// Maximum plies in a game or in a single MCTS rollout (avoids infinite loops).
pub const MAX_GAME_MOVES: u32 = 200;

/// When a straight-line continuation exists, use it with this probability; otherwise sample
/// uniformly over all legal moves (so continuation is never the only option).
const ROLLOUT_CONTINUATION_BIAS: f32 = 0.35;

/// Plays out a position to a terminal state and returns reward from `root_player`'s
/// perspective: +1 / -1 / 0.
pub trait RolloutPolicy {
    /// Evaluate a leaf position. Returns `(value, child_priors)`.
    ///
    /// `value` is the backup target in `[-1, 1]` from `root_player`'s perspective.
    /// `child_priors` is `Some(priors)` when the rollout policy has a learnt policy
    /// head (NN-based), or `None` for policies without one (random rollout → UCB1).
    fn rollout(&self, state: GameState, root_player: Player, rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>);

    /// Return policy priors for `state`'s legal actions without doing a full rollout.
    /// Used to initialise the root node before the search loop begins.
    /// Returns `None` for rollout policies without a policy network (→ UCB1 selection).
    fn priors_only(&self, _state: &GameState) -> Option<Vec<(Pos, f32)>> {
        None
    }

    /// If true, [`Mcts::search_iters`] may shard work across Rayon workers; each worker runs
    /// the same rollout type as the caller (which must be [`Send`] + [`Sync`]).
    const PARALLEL_SAFE: bool = false;
}

/// Tactical weight for placing at `pos` when `me` is to move.
///
/// Priority ladder:
///   100 — immediate win
///    80 — must block opponent's immediate win
///    20 — extends my run to 5
///    15 — blocks opponent's 5-in-a-row
///    12 — creates threats on 2+ axes (Triangle / Rhombus multi-axis attack)
/// Tactical prior weight for placing at `pos`.
///
/// Weights are intentionally lopsided toward defense: with the pair-move rule an
/// opponent 4-in-a-row is a near-certain win next turn (extend to 5, then 6 with
/// the second move of the pair), so blocking must dominate exploration.
///
///  1000 — win immediately
///   800 — block immediate loss (opp completes 6)
///   300 — block opp 5-in-a-row (they win with next move of pair)
///    60 — block opp 4-in-a-row (they go 4→5→6 next pair)
///    50 — create my own 5-in-a-row
///    30 — create multi-axis fork (Triangle/Rhombus)
///    25 — block opp multi-axis fork
///    10 — extend my run to 4
///     5 — create 3-in-a-row on any axis
///     4 — block opp 3-in-a-row
///     1 — normal move
///
/// Straight open-four endpoints plus any empty candidate where the opponent would reach a
/// run of ≥ 5 if they played there (same as compound `critical`, but cheap to precompute for
/// [`naive_best_move`]).
fn opp_merged_block_hints(state: &GameState, opp: Player) -> HashSet<Pos> {
    let mut s = opp_straight_extension_blocks(&state.board, opp);
    for &p in &state.candidates {
        if state.board.contains_key(&p) {
            continue;
        }
        if max_run_through(&state.board, p, opp) >= 5 {
            s.insert(p);
        }
    }
    s
}

/// `straight_opp_blocks` should be [`opp_straight_extension_blocks`] for this `board` and
/// `opp` (reuse one set per position when scoring many cells).
fn move_weight_core(
    board: &std::collections::HashMap<crate::game::Pos, crate::game::Player>,
    pos: crate::game::Pos,
    me: crate::game::Player,
    opp: crate::game::Player,
    straight_opp_blocks: &HashSet<Pos>,
) -> f32 {
    let my_runs = runs_per_axis(board, pos, me);
    let op_runs = runs_per_axis(board, pos, opp);
    let my_max = my_runs.iter().copied().max().unwrap_or(0);
    let op_max = op_runs.iter().copied().max().unwrap_or(0);
    let my_axes3 = my_runs.iter().filter(|&&r| r >= 3).count();
    let op_axes3 = op_runs.iter().filter(|&&r| r >= 3).count();
    let blocks_straight_four = straight_opp_blocks.contains(&pos);

    if my_max >= 6 {
        1000.0
    } else if op_max >= 6 {
        800.0
    } else if op_max >= 5 {
        300.0  // must block — opp wins with 2nd move of pair
    } else if op_max >= 4 || blocks_straight_four {
        200.0  // must block — opp goes 4→5→6 next pair (pair-move: one-pair kill)
    } else if my_max >= 5 {
        50.0
    } else if my_axes3 >= 2 {
        30.0
    } else if op_axes3 >= 2 {
        25.0
    } else if my_max >= 4 {
        10.0
    } else if my_axes3 >= 1 {
        5.0
    } else if op_axes3 >= 1 {
        4.0
    } else {
        1.0
    }
}

pub fn move_weight(
    board: &std::collections::HashMap<crate::game::Pos, crate::game::Player>,
    pos: crate::game::Pos,
    me: crate::game::Player,
    opp: crate::game::Player,
) -> f32 {
    let blocks = opp_straight_extension_blocks(board, opp);
    move_weight_core(board, pos, me, opp, &blocks)
}

/// Pick the best tactical move for the naive player using `move_weight` heuristics.
/// Returns `None` only when there are no legal actions.
pub fn naive_best_move(state: &GameState) -> Option<Pos> {
    let actions = state.legal_actions();
    if actions.is_empty() {
        return None;
    }
    let me = state.current_player();
    let opp = me.other();
    let block_hints = opp_merged_block_hints(state, opp);
    actions
        .iter()
        .copied()
        .max_by(|&a, &b| {
            move_weight_core(&state.board, a, me, opp, &block_hints)
                .partial_cmp(&move_weight_core(&state.board, b, me, opp, &block_hints))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Compute PUCT prior weights for `actions` with compound-threat awareness.
///
/// First scores every move with [`move_weight`] (single-position analysis), then
/// does a one-time board scan to count *five-extension-points*: empty cells where
/// `opp` placing would immediately create a 5-in-a-row (winning the same pair via
/// 5→6 on the second move).  With the pair-move rule, N such points means:
///
///   N == 0-1 → base weights are fine
///   N == 2   → both pair moves must block; suppress everything else
///   N >= 3   → can't block all five-extensions; still boost obvious **straight-four**
///              extension blocks — they are not in `critical` (which is opp run ≥ 5 at `pos`)
///              and were incorrectly squashed to 1.0, destroying PUCT.
pub fn compound_threat_priors(
    state: &GameState,
    actions: &[Pos],
    me: Player,
    opp: Player,
) -> Vec<(Pos, f32)> {
    // One-time board scan: positions where opp placing creates a run ≥ 5.
    let straight_blocks = opp_straight_extension_blocks(&state.board, opp);
    let critical: std::collections::HashSet<Pos> = state.candidates.iter()
        .filter(|&&p| !state.board.contains_key(&p))
        .filter(|&&p| max_run_through(&state.board, p, opp) >= 5)
        .cloned()
        .collect();
    let n = critical.len();

    /// When `n >= 3`, generic “critical” (opp five-extension) cells are noisy; **geometric**
    /// endpoints of an existing straight run of 4+ ([`opp_straight_extension_blocks`]) are a
    /// sharper signal and must outrank them so PUCT does not treat 50×300 ties as uniform.
    /// Must be **above** the own-five-threat tier (600) below: an open straight four still loses
    /// on the next pair if ignored, so it beats “start a 5-threat somewhere else”.
    const STRAIGHT_FOUR_BLOCK_PRIOR: f32 = 750.0;
    const CRITICAL_FIVE_EXT_PRIOR: f32 = 300.0;

    actions.iter().map(|&pos| {
        let base = move_weight_core(&state.board, pos, me, opp, &straight_blocks);
        let w = if n >= 3 {
            // Can't block all threats with one pair → offense-or-die.
            if base >= 1000.0 {
                base  // my immediate win
            } else if base >= 800.0 {
                base  // must still block opp's immediate win (op_max >= 6)
            } else if straight_blocks.contains(&pos) {
                STRAIGHT_FOUR_BLOCK_PRIOR
            } else if max_run_through(&state.board, pos, me) >= 5 {
                600.0 // create own 5-threat: forces opp to defend, races their win
            } else if critical.contains(&pos) {
                CRITICAL_FIVE_EXT_PRIOR
            } else {
                1.0
            }
        } else if n == 2 {
            // Both pair moves must block — suppress pure offense.
            if base >= 1000.0 {
                base
            } else if base >= 800.0 {
                base
            } else if max_run_through(&state.board, pos, me) >= 5 {
                // Own 5-threat forces opp to block, potentially nullifying their plan.
                base.max(500.0)
            } else if critical.contains(&pos) {
                base  // keep the 300 from move_weight
            } else if straight_blocks.contains(&pos) {
                base.max(STRAIGHT_FOUR_BLOCK_PRIOR)
            } else {
                1.0   // everything else is irrelevant this pair
            }
        } else {
            base  // 0 or 1 critical threat → per-move weights are sufficient
        };
        (pos, w)
    }).collect()
}

/// Raise prior weights before softmax so a few tactical moves (hundreds) do not drown in a sea
/// of weight-1.0 legal cells (~PUCT gets ~uniform 1% visit share each).
const TACTICAL_PRIOR_SHARPEN_POW: f32 = 2.35;

fn sharpen_normalize_priors(raw: Vec<(Pos, f32)>) -> Vec<(Pos, f32)> {
    let boosted: Vec<(Pos, f32)> = raw
        .into_iter()
        .map(|(p, w)| (p, w.max(1.0).powf(TACTICAL_PRIOR_SHARPEN_POW)))
        .collect();
    let sum: f32 = boosted.iter().map(|(_, w)| w).sum();
    boosted.into_iter().map(|(p, w)| (p, w / sum)).collect()
}

fn finalize_tactical_priors(raw: Vec<(Pos, f32)>, num_actions: usize) -> Vec<(Pos, f32)> {
    if num_actions >= 40 {
        sharpen_normalize_priors(raw)
    } else {
        let sum: f32 = raw.iter().map(|(_, w)| w).sum();
        raw.into_iter().map(|(p, w)| (p, w / sum)).collect()
    }
}

/// Uniform random playouts (used by the GUI and benchmarks).
pub struct RandomRollout;

/// Last two placements by one side; used to prefer “keep going” along `prev → last`.
#[derive(Default)]
struct LastTwoMoves {
    prev: Option<Pos>,
    last: Option<Pos>,
}

impl LastTwoMoves {
    fn record(&mut self, pos: Pos) {
        self.prev = self.last;
        self.last = Some(pos);
    }

    /// One step further in the same direction as `prev → last` (vector `last - prev`).
    fn continuation(&self) -> Option<Pos> {
        let (prev, last) = (self.prev?, self.last?);
        Some((last.0 + (last.0 - prev.0), last.1 + (last.1 - prev.1)))
    }
}

fn rollout_pick_action(
    state: &GameState,
    actions: &[Pos],
    x_hist: &LastTwoMoves,
    o_hist: &LastTwoMoves,
    rng: &mut impl Rng,
) -> Pos {
    let hist = match state.current_player() {
        Player::X => x_hist,
        Player::O => o_hist,
    };
    let continuation = hist.continuation().filter(|&c| actions.iter().any(|&a| a == c));
    if let Some(c) = continuation {
        if rng.gen::<f32>() < ROLLOUT_CONTINUATION_BIAS {
            return c;
        }
    }
    actions[rng.gen_range(0..actions.len())]
}

impl RolloutPolicy for RandomRollout {
    /// Compound-threat-aware priors so interactive play blocks/attacks correctly.
    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        let actions = state.legal_actions();
        if actions.is_empty() {
            return None;
        }
        let me = state.current_player();
        let opp = me.other();
        let raw = compound_threat_priors(state, &actions, me, opp);
        let max_w = raw.iter().map(|(_, w)| *w).fold(0.0f32, f32::max);
        if max_w <= 1.0 {
            return None; // all moves equal weight → let UCB1 do its thing
        }
        Some(finalize_tactical_priors(raw, actions.len()))
    }

    fn rollout(&self, mut state: GameState, root_player: Player, rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>) {
        let mut ply = 0u32;
        let mut x_hist = LastTwoMoves::default();
        let mut o_hist = LastTwoMoves::default();
        while !state.is_terminal() {
            if ply >= MAX_GAME_MOVES {
                return (state.board_heuristic(root_player), None);
            }
            let actions = state.legal_actions();
            if actions.is_empty() {
                break;
            }
            let action = rollout_pick_action(&state, &actions, &x_hist, &o_hist, rng);
            let who = state.current_player();
            state.place(action);
            match who {
                Player::X => x_hist.record(action),
                Player::O => o_hist.record(action),
            }
            ply += 1;
        }
        let value = match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => state.board_heuristic(root_player),
        };
        (value, None)
    }

    const PARALLEL_SAFE: bool = true;
}

/// Naive greedy opponent: always tries to extend its own longest run, ignores blocking.
///
/// Prior weight = 5^(run_length − 1), so extending a run of 3 (weight 25) dominates
/// a new isolated piece (weight 1). Used as a fixed sparring partner during self-play
/// to generate decisive games where real threats develop.
pub struct NaiveRollout;

impl RolloutPolicy for NaiveRollout {
    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        let actions = state.legal_actions();
        if actions.is_empty() {
            return None;
        }
        let me = state.current_player();
        let raw: Vec<(Pos, f32)> = actions
            .iter()
            .map(|&pos| {
                let run = max_run_through(&state.board, pos, me);
                let w = if run >= 6 { 1_000_000.0 } else { 5f32.powi(run as i32 - 1) };
                (pos, w)
            })
            .collect();
        let total: f32 = raw.iter().map(|(_, w)| w).sum();
        Some(raw.into_iter().map(|(p, w)| (p, w / total)).collect())
    }

    fn rollout(&self, mut state: GameState, root_player: Player, _rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>) {
        // Simulate greedily: always extend own longest run.
        let mut ply = 0u32;
        while !state.is_terminal() {
            if ply >= MAX_GAME_MOVES {
                return (state.board_heuristic(root_player), None);
            }
            let actions = state.legal_actions();
            if actions.is_empty() { break; }
            let me = state.current_player();
            let best = actions.iter().copied().max_by_key(|&pos| {
                max_run_through(&state.board, pos, me)
            }).unwrap_or(actions[0]);
            state.place(best);
            ply += 1;
        }
        let value = match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => state.board_heuristic(root_player),
        };
        (value, None)
    }
}

/// First this many plies of each simulation use [`naive_best_move`] (tactical); the rest
/// use the same light random + straight-line bias as [`RandomRollout`]. Shorter greedy
/// prefix keeps rollouts fast while still resolving urgent threats; the random tail reaches
/// more decisive terminals so backed-up Q values spread instead of clustering on
/// [`GameState::board_heuristic`].
pub const TACTICAL_ROLLOUT_GREEDY_PLIES: u32 = 14;

/// Heuristic MCTS: PUCT priors from [`compound_threat_priors`], hybrid rollouts (tactical
/// prefix + random tail). No NN — replay / analysis when nets are weak or unavailable.
pub struct TacticalRollout;

impl RolloutPolicy for TacticalRollout {
    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        let actions = state.legal_actions();
        if actions.is_empty() {
            return None;
        }
        let me = state.current_player();
        let opp = me.other();
        let raw = compound_threat_priors(state, &actions, me, opp);
        let max_w = raw.iter().map(|(_, w)| *w).fold(0.0f32, f32::max);
        if max_w <= 1.0 {
            return None;
        }
        Some(finalize_tactical_priors(raw, actions.len()))
    }

    fn rollout(&self, mut state: GameState, root_player: Player, rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>) {
        let mut ply = 0u32;
        let mut x_hist = LastTwoMoves::default();
        let mut o_hist = LastTwoMoves::default();
        while !state.is_terminal() {
            if ply >= MAX_GAME_MOVES {
                return (state.board_heuristic(root_player), None);
            }
            let actions = state.legal_actions();
            if actions.is_empty() {
                break;
            }
            let action = if ply < TACTICAL_ROLLOUT_GREEDY_PLIES {
                naive_best_move(&state).unwrap_or_else(|| actions[rng.gen_range(0..actions.len())])
            } else {
                rollout_pick_action(&state, &actions, &x_hist, &o_hist, rng)
            };
            let who = state.current_player();
            state.place(action);
            match who {
                Player::X => x_hist.record(action),
                Player::O => o_hist.record(action),
            }
            ply += 1;
        }
        let value = match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => state.board_heuristic(root_player),
        };
        (value, None)
    }

    const PARALLEL_SAFE: bool = true;
}

/// Sample an index into `unexpanded` proportional to prior mass on those moves.
/// Keeps first expansions aligned with PUCT; uniform fallback when all unexpanded priors are ~0.
fn sample_unexpanded_index_weighted(unexpanded: &[Pos], priors: &[(Pos, f32)], rng: &mut impl Rng) -> usize {
    let len = unexpanded.len();
    if len <= 1 {
        return 0;
    }
    let mut sum = 0.0f32;
    let mut weights = Vec::with_capacity(len);
    for &p in unexpanded {
        let w = priors
            .iter()
            .find(|(pp, _)| *pp == p)
            .map(|(_, pr)| *pr)
            .unwrap_or(0.0);
        sum += w;
        weights.push(w);
    }
    if sum <= 1e-12 {
        return rng.gen_range(0..len);
    }
    let t = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if t < acc || i + 1 == len {
            return i;
        }
    }
    len - 1
}

struct Node {
    state: GameState,
    action: Option<Pos>,
    parent: Option<usize>,
    children: Vec<usize>,
    /// Sum of simulation outcomes, from `root_player`'s perspective.
    total_value: f32,
    visits: u32,
    /// Actions not yet expanded into child nodes (random-ordered during expand).
    unexpanded: Vec<Pos>,
    /// Prior probability from parent's NN policy evaluation. 0.0 → fall back to UCB1.
    prior: f32,
    /// Cached NN policy priors for children (set on first NN evaluation of this node's state).
    children_priors: Option<Vec<(Pos, f32)>>,
}

pub struct Mcts {
    nodes: Vec<Node>,
    root_player: Player,
}

impl Mcts {
    pub fn new(state: GameState) -> Self {
        let root_player = state.current_player();
        let unexpanded = state.legal_actions();
        let root = Node {
            state,
            action: None,
            parent: None,
            children: vec![],
            total_value: 0.0,
            visits: 0,
            unexpanded,
            prior: 0.0,
            children_priors: None,
        };
        Mcts {
            nodes: vec![root],
            root_player,
        }
    }

    /// Run `n` MCTS iterations without returning results (used by self-play).
    ///
    /// Uses root-parallel MCTS across Rayon’s thread pool when `n` is large
    /// enough to use multiple workers; otherwise runs serially (same RNG).
    ///
    /// Parallel workers each run from an empty tree on a clone of the root
    /// [`GameState`], then statistics are merged additively into this tree.
    ///
    /// `P` must be [`Send`] + [`Sync`] so the same rollout policy can be shared across Rayon
    /// workers (e.g. [`RandomRollout`], [`TacticalRollout`]). Policies that opt out via
    /// [`RolloutPolicy::PARALLEL_SAFE`] still use this type parameter but run serially.
    pub fn search_iters<P: RolloutPolicy + Send + Sync>(
        &mut self,
        n: u32,
        rng: &mut impl Rng,
        rollout: &P,
    ) {
        // Initialise root children_priors for PUCT (no-op when rollout returns None priors).
        if self.nodes[0].children_priors.is_none() {
            if let Some(priors) = rollout.priors_only(&self.nodes[0].state) {
                self.nodes[0].children_priors = Some(priors);
            }
        }
        let threads = rayon::current_num_threads().max(1);
        if n == 0 {
            return;
        }
        // Require ~2 iters per Rayon worker so tree-clone cost pays off.
        if !P::PARALLEL_SAFE || threads == 1 || n < threads as u32 * 2 {
            self.search_iters_serial(n, rng, rollout);
        } else {
            self.search_iters_parallel(n, rollout);
        }
    }

    fn search_iters_serial<P: RolloutPolicy>(
        &mut self,
        n: u32,
        rng: &mut impl Rng,
        rollout: &P,
    ) {
        for _ in 0..n {
            let leaf = self.select(0);
            let child = self.expand(leaf, rng, rollout);
            let (reward, priors) = self.simulate(child, rng, rollout);
            if let Some(p) = priors {
                self.nodes[child].children_priors = Some(p);
            }
            self.backprop(child, reward);
        }
    }

    /// Root-parallel batch: each worker runs from an empty tree on a copy of
    /// the root [`GameState`], then root statistics are merged additively.
    /// Workers share the same `rollout` implementation (must be [`Send`] + [`Sync`]).
    fn search_iters_parallel<P: RolloutPolicy + Send + Sync>(&mut self, n: u32, rollout: &P) {
        let threads = rayon::current_num_threads().max(1);
        let num_workers = (threads as u32).min(n) as usize;
        let per = n / num_workers as u32;
        let rem = n % num_workers as u32;
        let chunks: Vec<u32> = (0..num_workers)
            .map(|i| per + if (i as u32) < rem { 1 } else { 0 })
            .filter(|&c| c > 0)
            .collect();

        let root_state = self.nodes[0].state.clone();

        let workers: Vec<Mcts> = chunks
            .into_par_iter()
            .map(|chunk| {
                let mut m = Mcts::new(root_state.clone());
                let mut rng = rand::thread_rng();
                m.search_iters_serial(chunk, &mut rng, rollout);
                m
            })
            .collect();

        for w in &workers {
            self.merge_fresh_worker_tree(w);
        }
    }

    /// Merge statistics from a worker tree that started empty at this root
    /// position (additive visits / total_value at root and root children).
    fn merge_fresh_worker_tree(&mut self, w: &Mcts) {
        self.nodes[0].visits += w.nodes[0].visits;
        self.nodes[0].total_value += w.nodes[0].total_value;

        for &wc in &w.nodes[0].children {
            let oc = &w.nodes[wc];
            let Some(pos) = oc.action else {
                continue;
            };
            if let Some(id) = self.find_root_child(pos) {
                self.nodes[id].visits += oc.visits;
                self.nodes[id].total_value += oc.total_value;
            } else {
                let new_id = self.copy_subtree_from(w, wc);
                self.nodes[new_id].parent = Some(0);
                self.nodes[0].children.push(new_id);
            }
        }
    }

    fn find_root_child(&self, pos: Pos) -> Option<usize> {
        self.nodes[0].children.iter().copied().find(|&cid| {
            self.nodes[cid].action == Some(pos)
        })
    }

    /// Deep-copy `other`’s subtree rooted at `root_id` into `self`’s arena.
    fn copy_subtree_from(&mut self, other: &Mcts, root_id: usize) -> usize {
        fn copy_node(this: &mut Mcts, other: &Mcts, oid: usize) -> usize {
            let node = &other.nodes[oid];
            let child_ids: Vec<usize> = node
                .children
                .iter()
                .map(|&c| copy_node(this, other, c))
                .collect();
            let new_node = Node {
                state: node.state.clone(),
                action: node.action,
                parent: None,
                children: child_ids.clone(),
                total_value: node.total_value,
                visits: node.visits,
                unexpanded: node.unexpanded.clone(),
                prior: node.prior,
                children_priors: node.children_priors.clone(),
            };
            let nid = this.nodes.len();
            this.nodes.push(new_node);
            for &cid in &child_ids {
                this.nodes[cid].parent = Some(nid);
            }
            nid
        }
        copy_node(self, other, root_id)
    }

    /// Return the top-`top_n` root children sorted by **descending visit count** (primary
    /// recommendation signal). Each entry is `(pos, value_display_0_1, visits, policy_share)`
    /// where `policy_share` is `visits / root_visits`.
    ///
    /// For **non-terminal** children, `value_display_0_1` is `(mean_backup + 1) / 2` with
    /// `mean_backup` the average MCTS backup in `[-1, 1]`. That is a **value index**, not a
    /// calibrated win probability — with heuristic or shallow rollouts it often clusters
    /// (use visits and this field together). For **terminal** children, the value is the
    /// exact outcome from [`Node::state`] for [`Self::root_player`].
    pub fn best_moves(&self, top_n: usize) -> Vec<(Pos, f32, u32, f32)> {
        let root_visits = self.nodes[0].visits;
        let root_children = self.nodes[0].children.clone();
        let mut results: Vec<(Pos, f32, u32, f32)> = root_children
            .iter()
            .filter_map(|&cid| {
                let n = &self.nodes[cid];
                let pos = n.action?;
                let win_rate = self.root_child_display_score(n);
                let policy_share = if root_visits > 0 {
                    n.visits as f32 / root_visits as f32
                } else {
                    0.0
                };
                Some((pos, win_rate, n.visits, policy_share))
            })
            .collect();
        // Sort by visit count (AlphaZero convention: most-visited = best recommendation).
        // Q-score (index 1) is unreliable when the value function is miscalibrated;
        // visit count is the robust signal — MCTS spends more effort where it's truly better.
        results.sort_by(|a, b| b.2.cmp(&a.2));
        results.truncate(top_n);
        results
    }

    /// Map stored backups to a `[0, 1]` display score; terminal positions use exact result.
    fn root_child_display_score(&self, n: &Node) -> f32 {
        if n.state.is_terminal() {
            return match n.state.winner {
                Some(p) if p == self.root_player => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
        }
        if n.visits > 0 {
            (n.total_value / n.visits as f32 + 1.0) / 2.0
        } else {
            0.5
        }
    }

    /// Total iterations run so far (= root visit count).
    pub fn total_visits(&self) -> u32 {
        self.nodes[0].visits
    }

    /// Return `(action, visit_count)` for every direct child of the root.
    pub fn root_children_stats(&self) -> Vec<(Pos, u32)> {
        self.nodes[0]
            .children
            .iter()
            .filter_map(|&cid| {
                let n = &self.nodes[cid];
                Some((n.action?, n.visits))
            })
            .collect()
    }

    /// Run `iterations` MCTS iterations and return all root children sorted by
    /// descending score (in [0,1]); see [`Self::best_moves`].
    /// Tuple is `(pos, score, visits, policy_share)`.
    pub fn search<P: RolloutPolicy>(
        &mut self,
        iterations: u32,
        rng: &mut impl Rng,
        rollout: &P,
    ) -> Vec<(Pos, f32, u32, f32)> {
        // Initialise root children_priors for PUCT (no-op for random rollout).
        if self.nodes[0].children_priors.is_none() {
            if let Some(priors) = rollout.priors_only(&self.nodes[0].state) {
                self.nodes[0].children_priors = Some(priors);
            }
        }
        for _ in 0..iterations {
            let leaf = self.select(0);
            let child = self.expand(leaf, rng, rollout);
            let (reward, priors) = self.simulate(child, rng, rollout);
            if let Some(p) = priors {
                self.nodes[child].children_priors = Some(p);
            }
            self.backprop(child, reward);
        }

        self.best_moves(usize::MAX)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Selection score for `node_id`.
    ///
    /// Uses PUCT (AlphaZero-style) when the node has a policy prior (`prior > 0`),
    /// otherwise falls back to UCB1 (random-rollout path).
    fn ucb(&self, node_id: usize) -> f32 {
        let node = &self.nodes[node_id];
        if node.visits == 0 {
            return f32::INFINITY;
        }
        let parent_id = match node.parent {
            Some(id) => id,
            None => return 0.0,
        };
        let parent = &self.nodes[parent_id];
        // Use root_player perspective: parent maximises if parent's player == root_player.
        let parent_player = parent.state.current_player();
        let q = if parent_player == self.root_player {
            node.total_value / node.visits as f32
        } else {
            -node.total_value / node.visits as f32
        };
        if node.prior > 0.0 {
            // PUCT: policy prior biases exploration toward moves the network prefers.
            q + C_PUCT * node.prior * (parent.visits as f32).sqrt() / (1.0 + node.visits as f32)
        } else {
            // UCB1: standard exploration when no policy priors are available.
            q + C * ((parent.visits as f32).ln() / node.visits as f32).sqrt()
        }
    }

    /// Walk down the tree using UCB until a node with unexpanded actions (or
    /// a terminal) is reached.
    fn select(&self, mut id: usize) -> usize {
        loop {
            let node = &self.nodes[id];
            if !node.unexpanded.is_empty() || node.state.is_terminal() {
                return id;
            }
            if node.children.is_empty() {
                return id;
            }
            id = *node
                .children
                .iter()
                .max_by(|&&a, &&b| self.ucb(a).total_cmp(&self.ucb(b)))
                .unwrap();
        }
    }

    /// Pick one unexpanded action, create a child node with its PUCT prior, return its id.
    fn expand<P: RolloutPolicy>(&mut self, id: usize, rng: &mut impl Rng, rollout: &P) -> usize {
        if self.nodes[id].state.is_terminal() {
            return id;
        }
        if self.nodes[id].unexpanded.is_empty() {
            return id;
        }

        // Lazily initialise children_priors for this node if not yet set.
        // This fires for the root on its first expansion; deeper nodes get their
        // children_priors set by the simulate return value.
        if self.nodes[id].children_priors.is_none() {
            if let Some(priors) = rollout.priors_only(&self.nodes[id].state) {
                self.nodes[id].children_priors = Some(priors);
            }
        }

        let idx = if let Some(ref priors) = self.nodes[id].children_priors {
            sample_unexpanded_index_weighted(&self.nodes[id].unexpanded, priors, rng)
        } else {
            rng.gen_range(0..self.nodes[id].unexpanded.len())
        };
        let action = self.nodes[id].unexpanded.swap_remove(idx);

        // Look up the NN policy prior for this action. 0.0 → UCB1 used in ucb().
        let prior = self.nodes[id]
            .children_priors
            .as_ref()
            .and_then(|ps| ps.iter().find(|(p, _)| *p == action).map(|(_, pr)| *pr))
            .unwrap_or(0.0);

        let mut new_state = self.nodes[id].state.clone();
        new_state.place(action);
        let unexpanded = new_state.legal_actions();

        let child = Node {
            state: new_state,
            action: Some(action),
            parent: Some(id),
            children: vec![],
            total_value: 0.0,
            visits: 0,
            unexpanded,
            prior,
            children_priors: None,
        };

        let child_id = self.nodes.len();
        self.nodes.push(child);
        self.nodes[id].children.push(child_id);
        child_id
    }

    /// Rollout from node `id`. Returns `(value, child_priors)`.
    ///
    /// For NN rollouts the returned priors should be stored in `nodes[id].children_priors`
    /// so subsequent expansions of that node can use PUCT selection.
    fn simulate<P: RolloutPolicy>(
        &self,
        id: usize,
        rng: &mut impl Rng,
        rollout: &P,
    ) -> (f32, Option<Vec<(Pos, f32)>>) {
        let state = self.nodes[id].state.clone();
        rollout.rollout(state, self.root_player, rng)
    }

    /// Propagate `reward` up to the root; every ancestor increments its visit
    /// counter and accumulates the same root-perspective reward value.
    fn backprop(&mut self, mut id: usize, reward: f32) {
        loop {
            self.nodes[id].visits += 1;
            self.nodes[id].total_value += reward;
            match self.nodes[id].parent {
                Some(parent_id) => id = parent_id,
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn last_two_moves_continues_straight_line() {
        let mut h = LastTwoMoves::default();
        h.record((0, 0));
        h.record((1, 0));
        assert_eq!(h.continuation(), Some((2, 0)));
    }

    #[test]
    fn search_iters_accumulates_root_visits() {
        let mut m = Mcts::new(GameState::new());
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100u32;
        let before = m.nodes[0].visits;
        let rollout = RandomRollout;
        m.search_iters(n, &mut rng, &rollout);
        assert_eq!(m.nodes[0].visits, before + n);
        assert_eq!(m.total_visits(), before + n);
    }

    #[test]
    fn parallel_merge_adds_onto_existing_root_visits() {
        let mut m = Mcts::new(GameState::new());
        let mut rng = StdRng::seed_from_u64(7);
        let rollout = RandomRollout;
        m.search_iters_serial(20, &mut rng, &rollout);
        let before = m.nodes[0].visits;
        m.search_iters(80, &mut rng, &rollout);
        assert_eq!(m.nodes[0].visits, before + 80);
    }

    /// X has 5 in a row on axis (1,0); playing (5,0) wins immediately (see `game` tests).
    fn state_x_one_move_wins() -> GameState {
        let moves = [
            (0, 0),
            (0, 20),
            (20, 0),
            (1, 0),
            (2, 0),
            (-20, 0),
            (0, -20),
            (3, 0),
            (4, 0),
            (20, 20),
            (-20, 20),
        ];
        let mut g = GameState::new();
        for &m in &moves {
            assert!(g.place(m), "illegal move {m:?}");
        }
        g
    }

    #[test]
    fn immediate_win_move_has_perfect_q_serial_and_parallel() {
        let g = state_x_one_move_wins();
        assert_eq!(g.current_player(), Player::X);
        assert!(g.legal_actions().contains(&(5, 0)));
        assert!(!g.is_terminal());

        let rollout = RandomRollout;
        let n = 8_000u32;

        let mut serial = Mcts::new(g.clone());
        let mut rng = StdRng::seed_from_u64(99);
        serial.search_iters_serial(n, &mut rng, &rollout);
        let q_serial = child_q(&serial, (5, 0));

        let mut parallel = Mcts::new(g);
        parallel.search_iters_parallel(n, &rollout);
        let q_parallel = child_q(&parallel, (5, 0));

        assert!(
            (q_serial - 1.0).abs() < 1e-3,
            "serial Q for winning move should be 1.0, got {q_serial}"
        );
        assert!(
            (q_parallel - 1.0).abs() < 1e-3,
            "parallel Q for winning move should be 1.0, got {q_parallel}"
        );
    }

    fn child_q(m: &Mcts, pos: Pos) -> f32 {
        let id = m
            .find_root_child(pos)
            .unwrap_or_else(|| panic!("no root child for {pos:?}"));
        let n = &m.nodes[id];
        assert!(n.visits > 0, "child {pos:?} never visited");
        n.total_value / n.visits as f32
    }

    #[test]
    fn block_straight_four_endpoint_outranks_scattered_development() {
        use std::collections::HashMap;
        let mut board = HashMap::new();
        for i in 1..=4 {
            board.insert((i, 0), Player::O);
        }
        board.insert((0, 8), Player::X);
        board.insert((1, 8), Player::X);
        board.insert((2, 8), Player::X);
        let blocks = crate::game::opp_straight_extension_blocks(&board, Player::O);
        let me = Player::X;
        let opp = Player::O;
        let w_block = move_weight_core(&board, (5, 0), me, opp, &blocks);
        let w_extend_self = move_weight_core(&board, (3, 8), me, opp, &blocks);
        assert!(
            w_block > w_extend_self,
            "must block O's open four before extending own shape: block {w_block} vs extend {w_extend_self}"
        );
        assert!(w_block >= 200.0, "endpoint block tier: {w_block}");
    }

    #[test]
    fn compound_threat_does_not_flatten_straight_four_when_n_ge_3() {
        // O has an open four on (1,0)..(4,0); X to move. Add three far-away X stones so O
        // has three distinct empty five-extension points (critical.len() >= 3).
        let cells = vec![
            ((1, 0), Player::O),
            ((2, 0), Player::O),
            ((3, 0), Player::O),
            ((4, 0), Player::O),
            ((0, 15), Player::X),
            ((1, 15), Player::X),
            ((2, 15), Player::X),
            ((3, 15), Player::X),
            ((0, -15), Player::X),
            ((1, -15), Player::X),
            ((2, -15), Player::X),
            ((3, -15), Player::X),
            ((15, 0), Player::X),
            ((16, 0), Player::X),
            ((17, 0), Player::X),
            ((18, 0), Player::X),
        ];
        let state = GameState::from_cells(&cells, cells.len() as u32);
        assert_eq!(state.current_player(), Player::X);
        let actions = state.legal_actions();
        let priors = compound_threat_priors(&state, &actions, Player::X, Player::O);
        let w_open_four_tip = priors
            .iter()
            .find(|(p, _)| *p == (5, 0))
            .map(|(_, w)| *w)
            .expect("(5,0) should be legal");
        assert!(
            w_open_four_tip > 10.0,
            "open-four extension must keep a strong prior when n>=3, got {w_open_four_tip}"
        );
    }

    /// Replay export: O open four on q=-8 with tip (-8,-2); X must block. Heuristics flag it;
    /// MCTS should concentrate visits on that move after prior-weighted expansion.
    #[test]
    fn replay_export_position_mcts_visits_block_most() {
        let cells: &[(Pos, Player)] = &[
            ((-9, 0), Player::O),
            ((-9, 1), Player::X),
            ((-9, 3), Player::O),
            ((-8, -1), Player::O),
            ((-8, 0), Player::O),
            ((-8, 1), Player::O),
            ((-8, 2), Player::O),
            ((-8, 3), Player::X),
            ((-7, -2), Player::X),
            ((-7, -1), Player::X),
            ((-7, 0), Player::X),
            ((-7, 1), Player::O),
            ((-5, 0), Player::X),
            ((-2, 0), Player::O),
            ((0, 0), Player::X),
            ((1, 0), Player::X),
            ((2, 0), Player::X),
            ((4, 0), Player::O),
            ((6, 0), Player::X),
            ((8, 0), Player::O),
        ];
        let state = GameState::from_cells(cells, 20);
        assert_eq!(state.current_player(), Player::X);
        let block = (-8, -2);
        assert!(state.legal_actions().contains(&block));

        let mut m = Mcts::new(state);
        let mut rng = StdRng::seed_from_u64(42);
        let rollout = TacticalRollout;
        m.search_iters_serial(6_000, &mut rng, &rollout);
        let top = m.best_moves(1);
        assert_eq!(
            top[0].0, block,
            "expected most-visited root child to be open-four block (-8,-2), got {:?} visits={}",
            top[0],
            top[0].2
        );
    }
}
