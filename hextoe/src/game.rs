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

pub fn check_win(board: &HashMap<Pos, Player>, pos: Pos, player: Player) -> bool {
    WIN_AXES
        .iter()
        .any(|&(dq, dr)| count_axis(board, pos, player, dq, dr) >= 6)
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

/// True when the next move is the first of a two-stone turn (not the anchor).
pub fn is_first_move_of_pair(state: &GameState) -> bool {
    state.total_moves > 0 && (state.total_moves - 1) % 2 == 0
}

/// Any legal move that immediately wins for the player to move (e.g. completing 6 in a row).
/// Only checks empty cells adjacent to `p` stones — a sixth stone must touch the line being extended.
pub fn find_immediate_win_move(state: &GameState) -> Option<Pos> {
    let p = state.current_player();
    let legal = state.legal_actions();
    let adj = empty_neighbors_of_player(&state.board, p);
    let to_try: Vec<Pos> = if adj.is_empty() {
        legal
    } else {
        adj
            .into_iter()
            .filter(|&pos| is_legal_move(state, pos))
            .collect()
    };
    for &pos in &to_try {
        let mut board = state.board.clone();
        board.insert(pos, p);
        if check_win(&board, pos, p) {
            return Some(pos);
        }
    }
    None
}

fn is_legal_move(state: &GameState, pos: Pos) -> bool {
    if state.winner.is_some() {
        return false;
    }
    if state.board.is_empty() {
        return pos == (0, 0);
    }
    state.candidates.contains(&pos)
}

/// Empty cells hex-adjacent to at least one stone of `player` (candidates for extending a line).
fn empty_neighbors_of_player(board: &HashMap<Pos, Player>, player: Player) -> Vec<Pos> {
    let mut v = Vec::new();
    for (&pos, &pl) in board.iter() {
        if pl != player {
            continue;
        }
        for &(dq, dr) in &HEX_DIRS {
            let n = (pos.0 + dq, pos.1 + dr);
            if !board.contains_key(&n) {
                v.push(n);
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// First stone of a two-move turn that can still win on the second stone (e.g. 4 in a row → 6).
/// Call only when no single legal move wins; assumes two stones remain this turn.
pub fn find_two_move_win_first(state: &GameState) -> Option<Pos> {
    let p = state.current_player();
    let legal = state.legal_actions();
    let adj = empty_neighbors_of_player(&state.board, p);
    let m1_list: Vec<Pos> = if adj.is_empty() {
        legal
    } else {
        let filtered: Vec<Pos> = adj
            .into_iter()
            .filter(|&pos| is_legal_move(state, pos))
            .collect();
        if filtered.is_empty() {
            return None;
        }
        filtered
    };
    for &m1 in &m1_list {
        let mut s = state.clone();
        if !s.place(m1) {
            continue;
        }
        if s.is_terminal() {
            continue;
        }
        let legal2 = s.legal_actions();
        let adj2 = empty_neighbors_of_player(&s.board, p);
        let m2_list: Vec<Pos> = if adj2.is_empty() {
            legal2
        } else {
            let filtered: Vec<Pos> = adj2
                .into_iter()
                .filter(|&pos| is_legal_move(&s, pos))
                .collect();
            if filtered.is_empty() {
                continue;
            }
            filtered
        };
        for &m2 in &m2_list {
            let mut s2 = s.clone();
            if !s2.place(m2) {
                continue;
            }
            if s2.winner == Some(p) {
                return Some(m1);
            }
        }
    }
    None
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
    fn immediate_win_detects_sixth_stone() {
        // Same line as win_with_six_in_row_axis0 but stop before (5,0); X must complete the line.
        let g = play(&[
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
        ]);
        assert_eq!(g.current_player(), Player::X);
        let w = find_immediate_win_move(&g).expect("winning move");
        assert!(
            w == (-1, 0) || w == (5, 0),
            "expected an endpoint completing six in a row, got {:?}",
            w
        );
    }

    #[test]
    fn two_move_win_first_extends_four_in_row() {
        // X at (0,0)..(3,0); second X of that pair is far so line stays at 4; (4,0),(5,0) empty.
        let g = play(&[
            (0, 0),
            (0, 20),
            (20, 0),
            (1, 0),
            (2, 0),
            (-20, 0),
            (0, -20),
            (3, 0),
            (100, 0),
            (20, 20),
            (-20, 20),
        ]);
        assert!(is_first_move_of_pair(&g));
        assert_eq!(find_immediate_win_move(&g), None);
        let m1 = find_two_move_win_first(&g).expect("two-move win");
        let mut s = g.clone();
        assert!(s.place(m1));
        assert!(!s.is_terminal());
        assert!(
            s.legal_actions().iter().any(|&m2| {
                let mut t = s.clone();
                t.place(m2) && t.winner == Some(Player::X)
            }),
            "second stone should complete a win after {:?}",
            m1
        );
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
