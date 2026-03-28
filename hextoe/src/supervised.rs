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

use crate::encode::{action_to_index, board_center, encode_state, GRID};
use crate::game::{GameState, Player, Pos};
use crate::mcts::compound_threat_priors;
use crate::nnue::encode_nnue;
use crate::self_play::GameRecord;
use crate::symmetry::{apply_transform, transform_state};

// ── Lightweight NNUE-only record ──────────────────────────────────────────────

/// Minimal training record for NNUE-only supervised pre-training.
/// Unlike [`GameRecord`] this does NOT store the CNN state encoding (~5 KB each),
/// so loading 9 000 games stays under ~1 GB even with 12× augmentation.
pub struct NnueRecord {
    pub feats: Vec<u16>,
    pub outcome: f32,
    /// Index into the GRID×GRID policy plane of the move that was played.
    /// `None` if the move fell outside the encoding window.
    pub move_idx: Option<u16>,
    /// Sparse normalized heuristic policy distribution over GRID×GRID positions.
    /// Only populated when `compound_threat_priors` returns a non-uniform distribution
    /// (i.e., there are actual threats on the board). Empty → no heuristic signal.
    pub heuristic_pi: Vec<(u16, f32)>,
}

/// Load one or more JSON files as [`NnueRecord`]s (no CNN state, no policy target).
///
/// Returns `(records, total_used, total_skipped)`.
pub fn load_nnue_records_multi(
    paths: &[String],
) -> Result<(Vec<NnueRecord>, usize, usize), Box<dyn std::error::Error>> {
    let mut all_records = Vec::new();
    let mut total_used = 0usize;
    let mut total_skipped = 0usize;
    for path in paths {
        let (records, used, skipped) = load_nnue_records(path)?;
        println!("  {path}: {used} games used, {skipped} skipped → {} positions", records.len());
        all_records.extend(records);
        total_used += used;
        total_skipped += skipped;
    }
    Ok((all_records, total_used, total_skipped))
}

fn load_nnue_records(
    path: &str,
) -> Result<(Vec<NnueRecord>, usize, usize), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let file: GamesFile = serde_json::from_str(&content)?;
    let mut records = Vec::new();
    let mut used = 0usize;
    let mut skipped = 0usize;
    for game in &file.games {
        match process_game_nnue(game) {
            Some(recs) => { records.extend(recs); used += 1; }
            None => skipped += 1,
        }
    }
    Ok((records, used, skipped))
}

fn process_game_nnue(game: &GameJson) -> Option<Vec<NnueRecord>> {
    if game.moves.is_empty() { return None; }
    let reason = game.game_result.reason.as_str();
    if !matches!(reason, "six-in-a-row" | "surrender") { return None; }
    let winner_id = game.game_result.winning_player_id.as_deref()?;

    let mut moves: Vec<&MoveJson> = game.moves.iter().collect();
    moves.sort_by_key(|m| m.move_number);

    let first_player_id = moves[0].player_id.as_str();
    let winner_player = if winner_id == first_player_id { Player::X } else { Player::O };

    let mut state = GameState::new();
    let mut steps: Vec<(GameState, Pos, Player)> = Vec::new();
    for m in &moves {
        if state.is_terminal() { break; }
        let pos = (m.x, m.y);
        let current_player = state.current_player();
        let snapshot = state.clone();
        if !state.place(pos) { return None; }
        steps.push((snapshot, pos, current_player));
    }
    if steps.is_empty() { return None; }

    let mut records = Vec::new();
    for (snap, pos, player) in &steps {
        let outcome = if *player == winner_player { 1.0f32 } else { -1.0f32 };
        for tid in 0u8..12 {
            let ts = transform_state(snap, tid);
            let tc = board_center(&ts);
            let feats: Vec<u16> = encode_nnue(&ts, tc).into_iter().map(|f| f as u16).collect();
            if !feats.is_empty() {
                let (tq, tr) = apply_transform(tid, pos.0, pos.1);
                let move_idx = action_to_index((tq, tr), tc).map(|i| i as u16);

                // Compute heuristic policy priors when there are actual threats.
                let heuristic_pi = {
                    let me = ts.current_player();
                    let opp = me.other();
                    let actions = ts.legal_actions();
                    if actions.is_empty() {
                        Vec::new()
                    } else {
                        let raw = compound_threat_priors(&ts, &actions, me, opp);
                        let max_w = raw.iter().map(|(_, w)| *w).fold(0.0f32, f32::max);
                        if max_w > 1.0 {
                            let total: f32 = raw.iter().map(|(_, w)| w).sum();
                            raw.into_iter()
                                .filter_map(|(p, w)| {
                                    action_to_index(p, tc).map(|i| (i as u16, w / total))
                                })
                                .collect()
                        } else {
                            Vec::new()
                        }
                    }
                };

                records.push(NnueRecord { feats, outcome, move_idx, heuristic_pi });
            }
        }
    }
    if records.is_empty() { None } else { Some(records) }
}

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

// ── RAM-efficient raw game loading ───────────────────────────────────────────

/// A game stored as a compact move sequence — no CNN encodings, no pre-augmentation.
/// Use with [`build_sample_index`] + [`encode_sample`] to encode on demand per batch.
pub struct RawGame {
    /// Full move sequence (coordinate pairs in play order).
    pub moves: Vec<Pos>,
    /// True if the first player (X) won.
    pub winner_is_first: bool,
}

/// Load games from one or more JSON files as lightweight [`RawGame`]s.
///
/// Returns `(games, total_used, total_skipped)`. Memory cost is roughly
/// `num_games × avg_moves × 8 bytes` — a few MB even for tens of thousands of games.
pub fn load_raw_games_multi(
    paths: &[String],
) -> Result<(Vec<RawGame>, usize, usize), Box<dyn std::error::Error>> {
    let mut all = Vec::new();
    let mut total_used = 0usize;
    let mut total_skipped = 0usize;
    for path in paths {
        let content = std::fs::read_to_string(path)?;
        let file: GamesFile = serde_json::from_str(&content)?;
        let (mut used, mut skipped) = (0, 0);
        for game in &file.games {
            match parse_raw_game(game) {
                Some(g) => { all.push(g); used += 1; }
                None => skipped += 1,
            }
        }
        println!("  {path}: {used} games used, {skipped} skipped");
        total_used += used;
        total_skipped += skipped;
    }
    Ok((all, total_used, total_skipped))
}

fn parse_raw_game(game: &GameJson) -> Option<RawGame> {
    if game.moves.is_empty() { return None; }
    let reason = game.game_result.reason.as_str();
    if !matches!(reason, "six-in-a-row" | "surrender") { return None; }
    let winner_id = game.game_result.winning_player_id.as_deref()?;

    let mut moves: Vec<&MoveJson> = game.moves.iter().collect();
    moves.sort_by_key(|m| m.move_number);

    let first_player_id = moves[0].player_id.as_str();
    let winner_is_first = winner_id == first_player_id;

    // Validate the game is playable.
    let mut state = GameState::new();
    for m in &moves {
        if state.is_terminal() { break; }
        if !state.place((m.x, m.y)) { return None; }
    }

    Some(RawGame {
        moves: moves.iter().map(|m| (m.x, m.y)).collect(),
        winner_is_first,
    })
}

/// Pre-compute all valid `(game_idx, step_idx)` pairs by replaying each game
/// once and checking that the played move falls inside the encoding window.
///
/// This index is tiny (~8 bytes per pair) and is shuffled for training.
pub fn build_sample_index(games: &[RawGame]) -> Vec<(usize, usize)> {
    let mut index = Vec::new();
    for (gi, game) in games.iter().enumerate() {
        let mut state = GameState::new();
        for (si, &pos) in game.moves.iter().enumerate() {
            if state.is_terminal() { break; }
            let center = board_center(&state);
            if action_to_index(pos, center).is_some() {
                index.push((gi, si));
            }
            state.place(pos);
        }
    }
    index
}

/// Encode a single `(game_idx, step_idx)` sample into a [`GameRecord`].
///
/// Replays the game to `step_idx`, applies symmetry transform `tid` (0–11),
/// then encodes. Returns `None` if the played move falls outside the grid after
/// the transform (rare edge case).
pub fn encode_sample(
    games: &[RawGame],
    game_idx: usize,
    step_idx: usize,
    tid: u8,
) -> Option<crate::self_play::GameRecord> {
    let game = &games[game_idx];
    let winner_player = if game.winner_is_first { Player::X } else { Player::O };

    // Replay to step_idx.
    let mut state = GameState::new();
    for &pos in &game.moves[..step_idx] {
        state.place(pos);
    }
    let player = state.current_player();
    let pos = game.moves[step_idx];
    let outcome = if player == winner_player { 1.0f32 } else { -1.0f32 };

    // Apply symmetry transform.
    let ts = transform_state(&state, tid);
    let tc = board_center(&ts);
    let tp = apply_transform(tid, pos.0, pos.1);
    let idx = action_to_index(tp, tc)?;

    let mut pi = [0.0f32; GRID * GRID];
    pi[idx] = 1.0;
    let state_enc = encode_state(&ts);

    Some(crate::self_play::GameRecord {
        state_enc: Box::new(state_enc),
        pi: Box::new(pi),
        outcome,
        nnue_feats: vec![],
        center: tc,
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load one or more JSON files and merge into a single record list.
///
/// Prints a per-file summary to stdout. Returns `(records, total_used, total_skipped)`.
pub fn load_supervised_records_multi(
    paths: &[String],
) -> Result<(Vec<GameRecord>, usize, usize), Box<dyn std::error::Error>> {
    let mut all_records = Vec::new();
    let mut total_used = 0usize;
    let mut total_skipped = 0usize;
    for path in paths {
        let (records, used, skipped) = load_supervised_records(path)?;
        println!("  {path}: {used} games used, {skipped} skipped → {} positions", records.len());
        all_records.extend(records);
        total_used += used;
        total_skipped += skipped;
    }
    Ok((all_records, total_used, total_skipped))
}

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

    // Replay the game, storing (state-before-move, pos, player) for later augmentation.
    let mut state = GameState::new();
    let mut steps: Vec<(GameState, Pos, Player)> = Vec::new();

    for m in &moves {
        if state.is_terminal() {
            break;
        }
        let pos = (m.x, m.y);
        let current_player = state.current_player();
        let snapshot = state.clone();
        if !state.place(pos) {
            // Illegal move — data is corrupt; discard the whole game.
            return None;
        }
        steps.push((snapshot, pos, current_player));
    }

    if steps.is_empty() {
        return None;
    }

    // Expand each step × 12 D₆ symmetry transforms, then encode.
    // This multiplies the training data by up to 12× for free.
    let mut records: Vec<GameRecord> = Vec::new();
    for (snap, pos, player) in &steps {
        let outcome = if *player == winner_player { 1.0f32 } else { -1.0f32 };
        for tid in 0u8..12 {
            let ts = transform_state(snap, tid);
            let tc = board_center(&ts);
            let tp: Pos = apply_transform(tid, pos.0, pos.1);

            // Skip if the (transformed) played move falls outside the encoding window.
            let Some(idx) = action_to_index(tp, tc) else { continue };
            let mut pi = [0.0f32; GRID * GRID];
            pi[idx] = 1.0;

            let state_enc = encode_state(&ts);
            let nnue_feats: Vec<u16> =
                encode_nnue(&ts, tc).into_iter().map(|f| f as u16).collect();

            records.push(GameRecord {
                state_enc: Box::new(state_enc),
                pi: Box::new(pi),
                outcome,
                nnue_feats,
                center: tc,
            });
        }
    }

    if records.is_empty() { None } else { Some(records) }
}
