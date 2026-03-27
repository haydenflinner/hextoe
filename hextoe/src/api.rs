//! Tiny REST API that receives moves from the browser bookmarklet.
//!
//! Listens on http://0.0.0.0:8080 and accepts:
//!   POST /game-state  — new session detected (with full board snapshot)
//!   POST /move        — single cell placed
//!   POST /game-over   — game finished (winner already applied via /move)
//!
//! Completed games are saved to `./live_games/<session_id>.json` in the same
//! format as the downloaded training data, so they can be fed straight into
//! `hextoe-pretrain` / `hextoe-pretrain-nnue`.

use std::sync::{mpsc, Arc, Mutex};

use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::game::{GameState, Player, Pos};

// ── Public update type sent to the GUI ────────────────────────────────────────

pub enum GameUpdate {
    /// Full board reset — replace the current game with this state.
    Reset(GameState),
    /// A single cell was placed at this position for `current_player()`.
    Move(Pos),
    /// A completed game was saved to disk; carries the file path.
    Saved(String),
}

// ── In-progress game recorder ─────────────────────────────────────────────────

struct PendingGame {
    session_id: String,
    x_id: String,
    o_id: String,
    /// Ordered moves accumulated from /move events.
    moves: Vec<SavedMove>,
    /// True only when we started recording from move 0 (or 1 if anchor was
    /// already in the /game-state snapshot).  Mid-game joins are not saved.
    complete_from_start: bool,
}

// ── JSON shapes — incoming ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CellJson {
    x: i32,
    y: i32,
    #[serde(rename = "occupiedBy")]
    occupied_by: String,
}

#[derive(Deserialize)]
struct PlayerJson {
    id: String,
}

#[derive(Deserialize)]
struct InnerState {
    cells: Vec<CellJson>,
    #[serde(rename = "currentTurnPlayerId")]
    current_turn_player_id: String,
}

#[derive(Deserialize)]
struct GameStateData {
    state: InnerState,
}

#[derive(Deserialize)]
struct GameStatePayload {
    #[serde(rename = "sessionId")]
    session_id: String,
    players: Vec<PlayerJson>,
    data: GameStateData,
}

#[derive(Deserialize)]
struct MoveCellJson {
    x: i32,
    y: i32,
    #[serde(rename = "occupiedBy")]
    occupied_by: String,
}

#[derive(Deserialize)]
struct MoveData {
    cell: MoveCellJson,
}

#[derive(Deserialize)]
struct MovePayload {
    data: MoveData,
}

#[derive(Deserialize)]
struct GameOverWinner {
    #[serde(rename = "playerId")]
    player_id: String,
}

#[derive(Deserialize)]
struct GameOverData {
    winner: Option<GameOverWinner>,
}

#[derive(Deserialize)]
struct GameOverPayload {
    #[serde(rename = "sessionId")]
    session_id: String,
    data: GameOverData,
}

// ── JSON shapes — outgoing (saved file) ───────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct SavedMove {
    #[serde(rename = "moveNumber")]
    move_number: u32,
    #[serde(rename = "playerId")]
    player_id: String,
    x: i32,
    y: i32,
}

#[derive(Serialize, Deserialize)]
struct SavedGameResult {
    #[serde(rename = "winningPlayerId")]
    winning_player_id: Option<String>,
    reason: String,
}

#[derive(Serialize, Deserialize)]
struct SavedGame {
    #[serde(rename = "gameResult")]
    game_result: SavedGameResult,
    moves: Vec<SavedMove>,
}

#[derive(Serialize, Deserialize)]
struct SavedGamesFile {
    games: Vec<SavedGame>,
}

// ── Axum shared state ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct ApiState {
    tx: Arc<mpsc::SyncSender<GameUpdate>>,
    pending: Arc<Mutex<Option<PendingGame>>>,
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn handle_game_state(
    State(api): State<ApiState>,
    Json(payload): Json<GameStatePayload>,
) -> StatusCode {
    let cells = &payload.data.state.cells;
    let current_turn_id = &payload.data.state.current_turn_player_id;
    let total_moves = cells.len() as u32;

    // X always moves first.  current_player(total_moves) tells us who goes next,
    // and that player's online ID == current_turn_player_id.
    let next_player = player_for_count(total_moves);

    let other_id = cells
        .iter()
        .find(|c| c.occupied_by != *current_turn_id)
        .map(|c| c.occupied_by.clone())
        .or_else(|| {
            payload.players
                .iter()
                .find(|p| p.id != *current_turn_id)
                .map(|p| p.id.clone())
        })
        .unwrap_or_default();

    let (x_id, o_id) = match next_player {
        Player::X => (current_turn_id.clone(), other_id),
        Player::O => (other_id, current_turn_id.clone()),
    };

    // Build the display state.
    let typed: Vec<(Pos, Player)> = cells
        .iter()
        .filter_map(|c| {
            let player = if c.occupied_by == x_id {
                Player::X
            } else if c.occupied_by == o_id {
                Player::O
            } else {
                return None;
            };
            Some(((c.x, c.y), player))
        })
        .collect();

    let state = GameState::from_cells(&typed, total_moves);
    let _ = api.tx.try_send(GameUpdate::Reset(state));

    // Start a new pending game.  We can only produce a complete record when we
    // catch the game from the very beginning (0 or 1 cells already placed).
    let complete_from_start = cells.len() <= 1;
    let initial_moves: Vec<SavedMove> = if cells.len() == 1 {
        // The anchor was already placed — record it as move 0.
        vec![SavedMove {
            move_number: 0,
            player_id: cells[0].occupied_by.clone(),
            x: cells[0].x,
            y: cells[0].y,
        }]
    } else {
        vec![]
    };

    *api.pending.lock().unwrap() = Some(PendingGame {
        session_id: payload.session_id,
        x_id,
        o_id,
        moves: initial_moves,
        complete_from_start,
    });

    StatusCode::OK
}

async fn handle_move(
    State(api): State<ApiState>,
    Json(payload): Json<MovePayload>,
) -> StatusCode {
    let pos = (payload.data.cell.x, payload.data.cell.y);
    let _ = api.tx.try_send(GameUpdate::Move(pos));

    // Append to the pending game recorder.
    if let Some(pending) = api.pending.lock().unwrap().as_mut() {
        if pending.complete_from_start {
            let move_number = pending.moves.len() as u32;
            pending.moves.push(SavedMove {
                move_number,
                player_id: payload.data.cell.occupied_by.clone(),
                x: payload.data.cell.x,
                y: payload.data.cell.y,
            });
        }
    }

    StatusCode::OK
}

async fn handle_game_over(
    State(api): State<ApiState>,
    Json(payload): Json<GameOverPayload>,
) -> StatusCode {
    let winner_id = payload.data.winner.as_ref().map(|w| w.player_id.clone());

    let save_result = {
        let pending = api.pending.lock().unwrap();
        if let Some(p) = pending.as_ref() {
            if p.complete_from_start
                && !p.moves.is_empty()
                && p.session_id == payload.session_id
            {
                Some(save_game(p, winner_id))
            } else {
                None
            }
        } else {
            None
        }
    };

    match save_result {
        Some(Ok(path)) => {
            let _ = api.tx.try_send(GameUpdate::Saved(path));
        }
        Some(Err(e)) => eprintln!("[HexBot] Failed to save game: {e}"),
        None => {}
    }

    // Clear the pending game — session is over.
    *api.pending.lock().unwrap() = None;

    StatusCode::OK
}

// ── Save logic ────────────────────────────────────────────────────────────────

fn save_game(pending: &PendingGame, winner_id: Option<String>) -> Result<String, Box<dyn std::error::Error>> {
    std::fs::create_dir_all("live_games")?;

    let path = format!("live_games/{}.json", pending.session_id);

    // If the file already exists (e.g. double game-over), append to its games array.
    let mut existing: SavedGamesFile = if std::path::Path::new(&path).exists() {
        let content = std::fs::read_to_string(&path)?;
        serde_json::from_str(&content).unwrap_or(SavedGamesFile { games: vec![] })
    } else {
        SavedGamesFile { games: vec![] }
    };

    let reason = if winner_id.is_some() { "six-in-a-row" } else { "surrender" }.to_string();
    existing.games.push(SavedGame {
        game_result: SavedGameResult { winning_player_id: winner_id, reason },
        moves: pending.moves.clone(),
    });

    std::fs::write(&path, serde_json::to_string_pretty(&existing)?)?;
    println!("[HexBot] Saved game → {path} ({} moves)", pending.moves.len());
    Ok(path)
}

// ── Server startup ────────────────────────────────────────────────────────────

/// Spawn the REST API server on port 8080 and return a receiver for GUI updates.
pub fn start_api_server() -> mpsc::Receiver<GameUpdate> {
    let (tx, rx) = mpsc::sync_channel(64);
    let tx = Arc::new(tx);
    let pending = Arc::new(Mutex::new(None));

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            let api_state = ApiState { tx, pending };
            let app = Router::new()
                .route("/game-state", post(handle_game_state))
                .route("/move", post(handle_move))
                .route("/game-over", post(handle_game_over))
                .with_state(api_state)
                .layer(CorsLayer::permissive());

            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
                .await
                .expect("bind :8080");
            println!("[HexBot] REST API on http://localhost:8080");
            axum::serve(listener, app).await.expect("axum serve");
        });
    });

    rx
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Who plays at move number `total_moves` (mirrors `GameState::current_player`).
fn player_for_count(total_moves: u32) -> Player {
    if total_moves == 0 {
        return Player::X;
    }
    let after_anchor = total_moves - 1;
    if (after_anchor / 2) % 2 == 0 {
        Player::O
    } else {
        Player::X
    }
}
