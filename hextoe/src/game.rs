use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Player {
    X,
    O,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }
}

pub type Pos = (i32, i32);

/// The 3 axes used for win-checking (each covers both directions).
const WIN_AXES: [Pos; 3] = [(1, 0), (0, 1), (1, -1)];

/// Six neighbours in axial hex coordinates.
pub const HEX_DIRS: [Pos; 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

#[derive(Clone, Debug)]
pub struct GameState {
    pub board: HashMap<Pos, Player>,
    /// All empty cells within hex-distance 2 of any placed piece.
    pub candidates: HashSet<Pos>,
    pub total_moves: u32,
    pub winner: Option<Player>,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: HashMap::new(),
            candidates: HashSet::new(),
            total_moves: 0,
            winner: None,
        }
    }

    /// Move pattern: X OO XX OO XX …
    /// move 0       → X  (anchor)
    /// moves 1,2    → O
    /// moves 3,4    → X
    /// moves 5,6    → O  …
    pub fn current_player(&self) -> Player {
        if self.total_moves == 0 {
            return Player::X;
        }
        let after_anchor = self.total_moves - 1;
        if (after_anchor / 2) % 2 == 0 {
            Player::O
        } else {
            Player::X
        }
    }

    /// Returns a human-readable turn descriptor for the UI.
    pub fn turn_label(&self) -> &'static str {
        if self.total_moves == 0 {
            "anchor"
        } else if (self.total_moves - 1) % 2 == 0 {
            "1st of 2"
        } else {
            "2nd of 2"
        }
    }

    /// Place on `pos` for the current player. Returns false if the move is illegal.
    pub fn place(&mut self, pos: Pos) -> bool {
        if self.board.contains_key(&pos) || self.winner.is_some() {
            return false;
        }
        let player = self.current_player();
        self.board.insert(pos, player);
        self.candidates.remove(&pos);
        self.expand_candidates(pos);
        self.total_moves += 1;
        if check_win(&self.board, pos, player) {
            self.winner = Some(player);
        }
        true
    }

    /// Legal moves: origin only on an empty board; otherwise all candidate cells.
    pub fn legal_actions(&self) -> Vec<Pos> {
        if self.winner.is_some() {
            return vec![];
        }
        if self.board.is_empty() {
            return vec![(0, 0)];
        }
        self.candidates.iter().copied().collect()
    }

    pub fn is_terminal(&self) -> bool {
        self.winner.is_some()
    }

    /// Add every hex cell within distance 2 of `pos` that is still empty.
    fn expand_candidates(&mut self, pos: Pos) {
        let (q, r) = pos;
        for dq in -2i32..=2 {
            let dr_lo = (-2).max(-dq - 2);
            let dr_hi = 2i32.min(-dq + 2);
            for dr in dr_lo..=dr_hi {
                let p = (q + dq, r + dr);
                if !self.board.contains_key(&p) {
                    self.candidates.insert(p);
                }
            }
        }
    }
}

/// Count consecutive `player` cells along axis `(dq, dr)` through `pos` (both directions).
fn count_axis(board: &HashMap<Pos, Player>, pos: Pos, player: Player, dq: i32, dr: i32) -> u32 {
    let mut count = 1;
    let (mut q, mut r) = (pos.0 + dq, pos.1 + dr);
    while board.get(&(q, r)) == Some(&player) {
        count += 1;
        q += dq;
        r += dr;
    }
    let (mut q, mut r) = (pos.0 - dq, pos.1 - dr);
    while board.get(&(q, r)) == Some(&player) {
        count += 1;
        q -= dq;
        r -= dr;
    }
    count
}

/// Maximum consecutive `player` run through `pos` across all three axes,
/// treating `pos` as if occupied by `player` (regardless of its actual state).
///
/// Use this to evaluate "how good would it be to place at `pos`?":
/// - returns ≥ 6 → placing here wins immediately
/// - returns 5 → extends to a 5-in-a-row (one step from winning)
/// - returns 4 → extends to a 4-in-a-row, etc.
pub fn max_run_through(board: &HashMap<Pos, Player>, pos: Pos, player: Player) -> u32 {
    WIN_AXES
        .iter()
        .map(|&(dq, dr)| count_axis(board, pos, player, dq, dr) + 1)
        .max()
        .unwrap_or(0)
}

pub fn check_win(board: &HashMap<Pos, Player>, pos: Pos, player: Player) -> bool {
    WIN_AXES
        .iter()
        .any(|&(dq, dr)| count_axis(board, pos, player, dq, dr) >= 6)
}

/// Exponentially-weighted threat score for one player: sum of `3^(k-1)` for each maximal
/// consecutive segment of length `k` across all three axes. A 5-in-a-row (81) heavily
/// outweighs scattered singles (1 each), matching the intuition that connected stones
/// are far more valuable than lonely ones.
fn threat_score(board: &HashMap<Pos, Player>, player: Player) -> f64 {
    let mut total = 0.0f64;
    for &(dq, dr) in &WIN_AXES {
        for &pos in board.keys() {
            if board.get(&pos) != Some(&player) {
                continue;
            }
            // Only process segments starting at this cell (predecessor is not ours).
            let prev = (pos.0 - dq, pos.1 - dr);
            if board.get(&prev) == Some(&player) {
                continue;
            }
            let mut len = 1u32;
            let (mut q, mut r) = (pos.0 + dq, pos.1 + dr);
            while board.get(&(q, r)) == Some(&player) {
                len += 1;
                q += dq;
                r += dr;
            }
            total += 3.0f64.powi(len as i32 - 1);
        }
    }
    total
}

impl GameState {
    /// Heuristic evaluation for a non-terminal position. Returns a value in `[-1, 1]` from
    /// `player`'s perspective, based on relative threat scores (how close each side is to
    /// building 6-in-a-row). Uses `tanh` so the signal is smooth and bounded.
    pub fn board_heuristic(&self, player: Player) -> f32 {
        let my = threat_score(&self.board, player);
        let opp = threat_score(&self.board, player.other());
        // Scale so a single 5-in-a-row advantage (~81 pts) maps to ~0.67.
        ((my - opp) / 100.0).tanh() as f32
    }
}

/// Return positions in the winning line through `pos` (used for highlight in GUI).
pub fn winning_line(board: &HashMap<Pos, Player>, pos: Pos, player: Player) -> Vec<Pos> {
    for &(dq, dr) in &WIN_AXES {
        if count_axis(board, pos, player, dq, dr) >= 6 {
            let mut line = vec![pos];
            let (mut q, mut r) = (pos.0 + dq, pos.1 + dr);
            while board.get(&(q, r)) == Some(&player) {
                line.push((q, r));
                q += dq;
                r += dr;
            }
            let (mut q, mut r) = (pos.0 - dq, pos.1 - dr);
            while board.get(&(q, r)) == Some(&player) {
                line.push((q, r));
                q -= dq;
                r -= dr;
            }
            return line;
        }
    }
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: play a sequence of moves (must be legal and in order).
    fn play(moves: &[(i32, i32)]) -> GameState {
        let mut g = GameState::new();
        for &m in moves {
            assert!(g.place(m), "illegal move {:?}", m);
        }
        g
    }

    #[test]
    fn move_order_xooxx() {
        // Pattern: X OO XX OO …
        let mut g = GameState::new();
        assert_eq!(g.current_player(), Player::X); // 0
        g.place((0, 0));
        assert_eq!(g.current_player(), Player::O); // 1
        g.place((10, 0));
        assert_eq!(g.current_player(), Player::O); // 2
        g.place((11, 0));
        assert_eq!(g.current_player(), Player::X); // 3
        g.place((1, 0));
        assert_eq!(g.current_player(), Player::X); // 4
        g.place((2, 0));
        assert_eq!(g.current_player(), Player::O); // 5
        g.place((12, 0));
        assert_eq!(g.current_player(), Player::O); // 6
    }

    #[test]
    fn no_win_with_five_in_row() {
        // 5 X's in a row → no win
        let g = play(&[
            (0, 0),   // X anchor
            (0, 5),   // O
            (0, 6),   // O
            (1, 0),   // X
            (2, 0),   // X
            (0, 7),   // O
            (0, 8),   // O
            (3, 0),   // X
            (4, 0),   // X  ← 5 in a row, not 6
        ]);
        assert_eq!(g.winner, None);
    }

    #[test]
    fn win_with_six_in_row_axis0() {
        // X gets 6 in a row along axis (1,0); O pieces are deliberately scattered.
        // Move order: X OO XX OO XX OO X
        let g = play(&[
            (0, 0),    // X – anchor
            (0, 20),   // O  (far away, isolated)
            (20, 0),   // O
            (1, 0),    // X
            (2, 0),    // X
            (-20, 0),  // O
            (0, -20),  // O
            (3, 0),    // X
            (4, 0),    // X
            (20, 20),  // O
            (-20, 20), // O
            (5, 0),    // X ← 6 in a row along (1,0)
        ]);
        assert_eq!(g.winner, Some(Player::X));
    }

    #[test]
    fn win_with_six_in_row_diagonal() {
        // X gets 6 in a row along axis (1,-1); O pieces are scattered.
        let g = play(&[
            (0, 0),    // X – anchor
            (0, 20),   // O
            (20, 0),   // O
            (1, -1),   // X
            (2, -2),   // X
            (-20, 0),  // O
            (0, -20),  // O
            (3, -3),   // X
            (4, -4),   // X
            (20, 20),  // O
            (-20, 20), // O
            (5, -5),   // X ← 6 in a row along (1,-1)
        ]);
        assert_eq!(g.winner, Some(Player::X));
    }

    #[test]
    fn no_self_overlap() {
        let mut g = GameState::new();
        assert!(g.place((0, 0)));
        assert!(!g.place((0, 0))); // duplicate → rejected
    }

    #[test]
    fn candidates_cover_neighbours() {
        let mut g = GameState::new();
        g.place((0, 0));
        // All 12 cells within distance 1-2 of origin should be candidates.
        let dist2: Vec<Pos> = {
            let mut v = vec![];
            for dq in -2i32..=2 {
                for dr in (-2i32).max(-dq - 2)..=2i32.min(-dq + 2) {
                    if dq != 0 || dr != 0 {
                        v.push((dq, dr));
                    }
                }
            }
            v
        };
        for p in dist2 {
            assert!(g.candidates.contains(&p), "{:?} not in candidates", p);
        }
    }
}
