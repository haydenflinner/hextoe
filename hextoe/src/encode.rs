use crate::game::{GameState, Player, Pos};

/// Side length of the square crop centred on the board centroid used for CNN input/output.
/// Radius = (GRID-1)/2 = 16 hexes from centre — covers >99% of competitive play.
pub const GRID: usize = 33;
pub const CHANNELS: usize = 4;

/// Return the rounded mean (q, r) of all board piece positions.
/// Returns (0, 0) on an empty board.
pub fn board_center(state: &GameState) -> (i32, i32) {
    if state.board.is_empty() {
        return (0, 0);
    }
    let n = state.board.len() as i32;
    let sum_q: i32 = state.board.keys().map(|&(q, _)| q).sum();
    let sum_r: i32 = state.board.keys().map(|&(_, r)| r).sum();
    // Rounded mean: add n/2 before integer division for round-half-up behaviour.
    let mean_q = if sum_q >= 0 {
        (sum_q + n / 2) / n
    } else {
        (sum_q - n / 2) / n
    };
    let mean_r = if sum_r >= 0 {
        (sum_r + n / 2) / n
    } else {
        (sum_r - n / 2) / n
    };
    (mean_q, mean_r)
}

/// Encode `state` into a flat `[CHANNELS * GRID * GRID]` tensor.
///
/// Layout: `[channel, row, col]` in row-major order.
/// Index formula: `(channel * GRID + row) * GRID + col`
///
/// Channels:
///   0 — Player::X piece positions
///   1 — Player::O piece positions
///   2 — current player's piece positions
///   3 — all 1.0 if current player is X, all 0.0 if O
pub fn encode_state(state: &GameState) -> [f32; CHANNELS * GRID * GRID] {
    let mut tensor = [0.0f32; CHANNELS * GRID * GRID];

    let center = board_center(state);
    let current = state.current_player();
    let half = (GRID / 2) as i32;

    // Channel 3: constant plane based on current player.
    if current == Player::X {
        for i in 0..GRID * GRID {
            tensor[3 * GRID * GRID + i] = 1.0;
        }
    }

    // Channels 0, 1, 2: iterate over board pieces.
    for (&(q, r), &player) in &state.board {
        let row = r - center.1 + half;
        let col = q - center.0 + half;
        if row < 0 || row >= GRID as i32 || col < 0 || col >= GRID as i32 {
            continue;
        }
        let row = row as usize;
        let col = col as usize;

        let ch0 = if player == Player::X { 0 } else { 1 };
        tensor[(ch0 * GRID + row) * GRID + col] = 1.0;

        if player == current {
            tensor[(2 * GRID + row) * GRID + col] = 1.0;
        }
    }

    tensor
}

/// Map an axial position to a flat grid index (`row * GRID + col`).
/// Returns `None` if the position falls outside the `GRID x GRID` window.
pub fn action_to_index(pos: Pos, center: (i32, i32)) -> Option<usize> {
    let half = (GRID / 2) as i32;
    let row = pos.1 - center.1 + half;
    let col = pos.0 - center.0 + half;
    if row < 0 || row >= GRID as i32 || col < 0 || col >= GRID as i32 {
        return None;
    }
    Some(row as usize * GRID + col as usize)
}

/// Convert a flat grid index back to an axial position.
pub fn index_to_action(idx: usize, center: (i32, i32)) -> Pos {
    let half = (GRID / 2) as i32;
    let row = (idx / GRID) as i32;
    let col = (idx % GRID) as i32;
    let q = col + center.0 - half;
    let r = row + center.1 - half;
    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    #[test]
    fn center_empty_board() {
        let state = GameState::new();
        assert_eq!(board_center(&state), (0, 0));
    }

    #[test]
    fn encode_empty_board_all_zeros_except_ch3() {
        let state = GameState::new();
        let tensor = encode_state(&state);
        // Channels 0-2 should all be zero.
        for i in 0..3 * GRID * GRID {
            assert_eq!(tensor[i], 0.0, "expected 0 at index {}", i);
        }
        // Channel 3 should be all 1.0 (first player is X).
        for i in 3 * GRID * GRID..4 * GRID * GRID {
            assert_eq!(tensor[i], 1.0, "expected 1 at index {}", i);
        }
    }

    #[test]
    fn roundtrip_action_index() {
        let center = (0, 0);
        let pos = (3, -2);
        let idx = action_to_index(pos, center).unwrap();
        assert_eq!(index_to_action(idx, center), pos);
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let center = (0, 0);
        let far = (100, 100);
        assert!(action_to_index(far, center).is_none());
    }
}
