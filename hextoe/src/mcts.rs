/// Specialized MCTS for Hextoe.
///
/// Key differences from the generic mcts-rs reference:
/// - Node arena uses `Vec<Node>` with index-based links (no Box/Rc).
/// - `GameState` maintains an incremental `candidates` set, so legal-action
///   enumeration never rescans the full board.
/// - Rewards are stored from a fixed root-player perspective, avoiding
///   per-level sign bookkeeping that breaks for 2-moves-per-turn games.
use crate::game::{GameState, Player, Pos};
use rand::Rng;
use rayon::prelude::*;

const C: f32 = std::f32::consts::SQRT_2;

/// Maximum plies in a game or in a single MCTS rollout (avoids infinite loops).
pub const MAX_GAME_MOVES: u32 = 100;

/// When a straight-line continuation exists, use it with this probability; otherwise sample
/// uniformly over all legal moves (so continuation is never the only option).
const ROLLOUT_CONTINUATION_BIAS: f32 = 0.35;

/// Plays out a position to a terminal state and returns reward from `root_player`'s
/// perspective: +1 / -1 / 0.
pub trait RolloutPolicy {
    fn rollout(&self, state: GameState, root_player: Player, rng: &mut impl Rng) -> f32;

    /// If true, root-parallel search may use an equivalent stateless rollout
    /// ([`RandomRollout`] only). Other policies always run serially.
    const PARALLEL_SAFE: bool = false;
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
    fn rollout(&self, mut state: GameState, root_player: Player, rng: &mut impl Rng) -> f32 {
        let mut ply = 0u32;
        let mut x_hist = LastTwoMoves::default();
        let mut o_hist = LastTwoMoves::default();
        while !state.is_terminal() {
            if ply >= MAX_GAME_MOVES {
                return state.board_heuristic(root_player);
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
        match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => state.board_heuristic(root_player),
        }
    }

    const PARALLEL_SAFE: bool = true;
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
    pub fn search_iters<P: RolloutPolicy>(&mut self, n: u32, rng: &mut impl Rng, rollout: &P) {
        let threads = rayon::current_num_threads().max(1);
        if n == 0 {
            return;
        }
        // Require ~2 iters per Rayon worker so tree-clone cost pays off.
        // NN / custom rollouts always run serially (parallel workers use [`RandomRollout`] only).
        if !P::PARALLEL_SAFE || threads == 1 || n < threads as u32 * 2 {
            self.search_iters_serial(n, rng, rollout);
        } else {
            self.search_iters_parallel(n);
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
            let child = self.expand(leaf, rng);
            let reward = self.simulate(child, rng, rollout);
            self.backprop(child, reward);
        }
    }

    /// Root-parallel batch: each worker runs from an empty tree on a copy of
    /// the root [`GameState`], then root statistics are merged additively.
    /// (Workers use [`rand::thread_rng`] and [`RandomRollout`].)
    fn search_iters_parallel(&mut self, n: u32) {
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
                let rollout = RandomRollout;
                m.search_iters_serial(chunk, &mut rng, &rollout);
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

    /// Return the top-`top_n` root children sorted by descending score.
    /// Each entry is `(pos, score_0_1, visits, policy_share)` where `policy_share`
    /// is `visits / root_visits` (fraction of simulations that took that edge from root).
    ///
    /// For **non-terminal** children, `score_0_1` is `(mean_backup + 1) / 2` with
    /// `mean_backup` the average MCTS backup target in `[-1, 1]` (random rollout
    /// outcomes and/or neural value at the leaf). That is an **estimate**, not an
    /// exact win probability unless rollouts are unbiased playouts to terminal.
    /// For **terminal** children, the score is the exact outcome from [`Node::state`]
    /// (win / loss / draw for [`Self::root_player`]), ignoring any float noise in
    /// accumulated `total_value`.
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
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
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
        for _ in 0..iterations {
            let leaf = self.select(0);
            let child = self.expand(leaf, rng);
            let reward = self.simulate(child, rng, rollout);
            self.backprop(child, reward);
        }

        self.best_moves(usize::MAX)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// UCB1 value of `node_id` from the perspective of its parent's player.
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
        q + C * ((parent.visits as f32).ln() / node.visits as f32).sqrt()
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

    /// Pick one unexpanded action at random, create a child node, return its id.
    fn expand(&mut self, id: usize, rng: &mut impl Rng) -> usize {
        if self.nodes[id].state.is_terminal() {
            return id;
        }
        if self.nodes[id].unexpanded.is_empty() {
            return id;
        }

        let idx = rng.gen_range(0..self.nodes[id].unexpanded.len());
        let action = self.nodes[id].unexpanded.swap_remove(idx);

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
        };

        let child_id = self.nodes.len();
        self.nodes.push(child);
        self.nodes[id].children.push(child_id);
        child_id
    }

    /// Rollout from node `id` using `rollout`; returns +1 (root wins) / -1 / 0.
    fn simulate<P: RolloutPolicy>(
        &self,
        id: usize,
        rng: &mut impl Rng,
        rollout: &P,
    ) -> f32 {
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
        parallel.search_iters_parallel(n);
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
}
