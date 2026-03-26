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

const C: f32 = std::f32::consts::SQRT_2;

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

    /// Run `iterations` MCTS iterations and return all root children sorted by
    /// descending estimated win-rate (in [0,1]) from root_player's perspective.
    pub fn search(&mut self, iterations: u32, rng: &mut impl Rng) -> Vec<(Pos, f32)> {
        for _ in 0..iterations {
            let leaf = self.select(0);
            let child = self.expand(leaf, rng);
            let reward = self.simulate(child, rng);
            self.backprop(child, reward);
        }

        let root_children = self.nodes[0].children.clone();
        let mut results: Vec<(Pos, f32)> = root_children
            .iter()
            .filter_map(|&cid| {
                let n = &self.nodes[cid];
                let pos = n.action?;
                let win_rate = if n.visits > 0 {
                    (n.total_value / n.visits as f32 + 1.0) / 2.0
                } else {
                    0.5
                };
                Some((pos, win_rate))
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
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
                .max_by(|&&a, &&b| {
                    self.ucb(a)
                        .partial_cmp(&self.ucb(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
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

    /// Random rollout from node `id`; returns +1 (root wins) / -1 / 0.
    fn simulate(&self, id: usize, rng: &mut impl Rng) -> f32 {
        let mut state = self.nodes[id].state.clone();
        while !state.is_terminal() {
            let actions = state.legal_actions();
            if actions.is_empty() {
                break;
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state.place(action);
        }
        match state.winner {
            Some(p) if p == self.root_player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
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
