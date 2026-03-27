//! D₆ symmetry group for the hex grid.
//!
//! The hex grid with a fixed center piece at (0,0) has dihedral symmetry D₆:
//! 6 rotations × 1 + 6 reflections = 12 transforms total. All preserve (0,0),
//! so they are always valid regardless of board size.
//!
//! Transform IDs:
//!   0–5  : rotations (0°, 60°, …, 300° CCW)
//!   6–11 : reflection across q-axis, then rotate 0°…300° CCW

use crate::game::GameState;

// ── Primitives ────────────────────────────────────────────────────────────────

/// Rotate a hex position 60° CCW in axial coordinates.
#[inline]
pub fn rotate60(q: i32, r: i32) -> (i32, i32) {
    (-r, q + r)
}

/// Reflect across the q-axis (swaps r ↔ s = −q−r).
#[inline]
pub fn reflect_q(q: i32, r: i32) -> (i32, i32) {
    (q, -r - q)
}

/// Apply one of the 12 D₆ transforms to a hex position.
///
/// - `tid` 0–5: pure CCW rotations (0°, 60°, …, 300°).
/// - `tid` 6–11: reflect across q-axis first, then rotate (tid−6) × 60°.
#[inline]
pub fn apply_transform(tid: u8, q: i32, r: i32) -> (i32, i32) {
    let (mut q, mut r) = if tid >= 6 { reflect_q(q, r) } else { (q, r) };
    let rots = tid % 6;
    for _ in 0..rots {
        (q, r) = rotate60(q, r);
    }
    (q, r)
}

/// Inverse of `apply_transform`: `apply_transform(inverse(tid), …) ∘ apply_transform(tid, …) = id`.
///
/// Rotations: inverse of r^k is r^(6-k).
/// Reflections: s·r^k is its own inverse (applying it twice = identity).
#[inline]
pub fn inverse_transform(tid: u8) -> u8 {
    if tid < 6 {
        (6 - tid) % 6
    } else {
        tid // s·r^k is self-inverse for all k
    }
}

// ── GameState transform ───────────────────────────────────────────────────────

/// Return a copy of `state` with all piece/candidate positions transformed by `tid`.
///
/// The hex distance metric is invariant under D₆, so:
/// - the board dictionary maps T(pos) → same player
/// - the candidates set (empty cells within distance 2 of any piece) is exactly
///   {T(c) | c ∈ original candidates}
pub fn transform_state(state: &GameState, tid: u8) -> GameState {
    if tid == 0 {
        return state.clone();
    }
    let new_board = state
        .board
        .iter()
        .map(|(&(q, r), &player)| (apply_transform(tid, q, r), player))
        .collect();
    let new_candidates = state
        .candidates
        .iter()
        .map(|&(q, r)| apply_transform(tid, q, r))
        .collect();
    GameState {
        board: new_board,
        candidates: new_candidates,
        total_moves: state.total_moves,
        winner: state.winner,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate60_six_times_is_identity() {
        let p = (3, -1);
        let mut q = p;
        for _ in 0..6 {
            q = rotate60(q.0, q.1);
        }
        assert_eq!(q, p);
    }

    #[test]
    fn reflect_twice_is_identity() {
        let p = (2, -5);
        assert_eq!(reflect_q(reflect_q(p.0, p.1).0, reflect_q(p.0, p.1).1), p);
    }

    #[test]
    fn all_12_transforms_are_distinct() {
        let p = (1, 2);
        let imgs: Vec<_> = (0u8..12).map(|t| apply_transform(t, p.0, p.1)).collect();
        // All 12 images of (1,2) under D₆ are distinct (generic point)
        let unique: std::collections::HashSet<_> = imgs.iter().copied().collect();
        assert_eq!(unique.len(), 12);
    }

    #[test]
    fn inverse_undoes_transform() {
        let p = (3, -2);
        for tid in 0u8..12 {
            let (tq, tr) = apply_transform(tid, p.0, p.1);
            let inv = inverse_transform(tid);
            assert_eq!(apply_transform(inv, tq, tr), p, "failed at tid={tid}");
        }
    }

    #[test]
    fn transform_state_preserves_piece_count() {
        use crate::game::GameState;
        let mut s = GameState::new();
        s.place((0, 0));
        s.place((1, 0));
        s.place((0, 1));
        for tid in 0u8..12 {
            let t = transform_state(&s, tid);
            assert_eq!(t.board.len(), s.board.len(), "tid={tid}");
            assert_eq!(t.candidates.len(), s.candidates.len(), "tid={tid}");
        }
    }

    #[test]
    fn transform_state_preserves_disjoint_board_candidates() {
        use crate::game::GameState;
        let mut s = GameState::new();
        for pos in [(0, 0), (1, 0), (0, 1), (-1, 1), (2, -1)] {
            s.place(pos);
        }
        for tid in 0u8..12 {
            let t = transform_state(&s, tid);
            for pos in t.board.keys() {
                assert!(!t.candidates.contains(pos), "tid={tid}: {pos:?} in both");
            }
        }
    }
}
