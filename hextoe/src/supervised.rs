//! Load online game records (JSON) and convert them to [`GameRecord`]s for training.
//!
//! Expected JSON schema (from the hextoe web scraper):
//! ```json
//! { "games": [ { "players": [...], "gameResult": {...}, "moves": [...] } ] }
//! ```
//!
//! Only games with reason `"six-in-a-row"` or `"surrender"` are used (clear winner).
//! Moves with coordinates outside the GRID encoding window are silently skipped.

use serde::Deserialize;

use crate::encode::{action_to_index, board_center, encode_state, CHANNELS, GRID};
use crate::game::{GameState, Player};
use crate::nnue::encode_nnue;
use crate::self_play::GameRecord;

// ── JSON schema ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MoveJson {
    #[serde(rename = "moveNumber")]
    move_number: u32,
    #[serde(rename = "playerId")]
    player_id: String,
    x: i32,
    y: i32,
}

#[derive(Deserialize)]
struct GameResultJson {
    #[serde(rename = "winningPlayerId")]
    winning_player_id: Option<String>,
    reason: String,
}

#[derive(Deserialize)]
struct GameJson {
    #[serde(rename = "gameResult")]
    game_result: GameResultJson,
    moves: Vec<MoveJson>,
}

#[derive(Deserialize)]
struct GamesFile {
    games: Vec<GameJson>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a JSON file of online games and return [`GameRecord`]s ready for training.
///
/// Each position in each accepted game becomes one record:
/// - `state_enc` — encoded board at the time of the move
/// - `pi`        — one-hot over the GRID²-action space at the played move index
/// - `outcome`   — `+1` if the player to move won, `-1` if they lost
///
/// Returns `(records, games_used, games_skipped)`.
pub fn load_supervised_records(
    path: &str,
) -> Result<(Vec<GameRecord>, usize, usize), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let file: GamesFile = serde_json::from_str(&content)?;

    let mut records: Vec<GameRecord> = Vec::new();
    let mut games_used = 0usize;
    let mut games_skipped = 0usize;

    for game in &file.games {
        match process_game(game) {
            Some(recs) => {
                records.extend(recs);
                games_used += 1;
            }
            None => games_skipped += 1,
        }
    }

    Ok((records, games_used, games_skipped))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn process_game(game: &GameJson) -> Option<Vec<GameRecord>> {
    if game.moves.is_empty() {
        return None;
    }

    // Only use games with a clear winner.
    let reason = game.game_result.reason.as_str();
    if !matches!(reason, "six-in-a-row" | "surrender") {
        return None;
    }
    let winner_id = game.game_result.winning_player_id.as_deref()?;

    // Sort moves by move number (they're usually ordered but let's be safe).
    let mut moves: Vec<&MoveJson> = game.moves.iter().collect();
    moves.sort_by_key(|m| m.move_number);

    // The first player in the move list is X in our GameState.
    let first_player_id = moves[0].player_id.as_str();
    let winner_is_first = winner_id == first_player_id;
    let winner_player = if winner_is_first { Player::X } else { Player::O };

    // Replay the game, recording (state_enc, pi, current_player) at each move.
    let mut state = GameState::new();
    // steps: (state_enc, pi, player_to_move)
    let mut steps: Vec<([f32; CHANNELS * GRID * GRID], [f32; GRID * GRID], Player, Vec<u16>)> =
        Vec::new();

    for m in &moves {
        if state.is_terminal() {
            break;
        }

        let pos = (m.x, m.y);
        let center = board_center(&state);
        let current_player = state.current_player();

        // Build one-hot pi. Skip positions where the move falls outside the encoding window.
        let mut pi = [0.0f32; GRID * GRID];
        if let Some(idx) = action_to_index(pos, center) {
            pi[idx] = 1.0;
            let state_enc = encode_state(&state);
            let nnue_feats: Vec<u16> =
                encode_nnue(&state, center).into_iter().map(|f| f as u16).collect();
            steps.push((state_enc, pi, current_player, nnue_feats));
        }
        // Whether or not we recorded this step, always advance the game state.
        if !state.place(pos) {
            // Illegal move — data is corrupt; discard the whole game.
            return None;
        }
    }

    if steps.is_empty() {
        return None;
    }

    // Convert (state_enc, pi, player) → GameRecord with outcome from player's perspective.
    Some(
        steps
            .into_iter()
            .map(|(state_enc, pi, player, nnue_feats)| {
                let outcome = if player == winner_player { 1.0f32 } else { -1.0f32 };
                GameRecord {
                    state_enc: Box::new(state_enc),
                    pi: Box::new(pi),
                    outcome,
                    nnue_feats,
                }
            })
            .collect(),
    )
}
