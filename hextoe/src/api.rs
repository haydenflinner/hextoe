//! Tiny REST API that receives moves from the browser bookmarklet.
//!
//! Listens on http://0.0.0.0:8080 and accepts:
//!   POST /game-state  — new session detected (with full board snapshot)
//!   POST /move        — single cell placed
//!   POST /game-over   — game finished (winner already applied via /move)

use std::sync::{mpsc, Arc, Mutex};

use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;
use serde::Deserialize;
use tower_http::cors::CorsLayer;

use crate::game::{check_win, GameState, Player, Pos};

// ── Public update type sent to the GUI ────────────────────────────────────────

pub enum GameUpdate {
    /// Full board reset — replace the current game with this state.
    Reset(GameState),
    /// A single cell was placed at this position for `current_player()`.
    Move(Pos),
}

// ── JSON shapes ───────────────────────────────────────────────────────────────

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
    players: Vec<PlayerJson>,
    data: GameStateData,
}

#[derive(Deserialize)]
struct MoveCell {
    x: i32,
    y: i32,
}

#[derive(Deserialize)]
struct MoveData {
    cell: MoveCell,
}

#[derive(Deserialize)]
struct MovePayload {
    data: MoveData,
}

// ── Axum shared state ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct ApiState {
    tx: Arc<mpsc::SyncSender<GameUpdate>>,
    /// Maps online player IDs to X / O.  Set on /game-state.
    player_map: Arc<Mutex<Option<(String, String)>>>, // (x_id, o_id)
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

    // The other ID: look for a cell owner != current_turn_id, or fall back to players[].
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

    // Store the mapping for future /move events.
    *api.player_map.lock().unwrap() = Some((x_id.clone(), o_id.clone()));

    // Build typed cell list.
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
    StatusCode::OK
}

async fn handle_move(
    State(api): State<ApiState>,
    Json(payload): Json<MovePayload>,
) -> StatusCode {
    let pos = (payload.data.cell.x, payload.data.cell.y);
    let _ = api.tx.try_send(GameUpdate::Move(pos));
    StatusCode::OK
}

async fn handle_game_over(
    State(_api): State<ApiState>,
    Json(_body): Json<serde_json::Value>,
) -> StatusCode {
    // The winning move was already delivered via /move, so the GUI's game state
    // already reflects the winner.  Nothing extra needed.
    StatusCode::OK
}

// ── Server startup ────────────────────────────────────────────────────────────

/// Spawn the REST API server on port 8080 and return a receiver for GUI updates.
pub fn start_api_server() -> mpsc::Receiver<GameUpdate> {
    let (tx, rx) = mpsc::sync_channel(64);
    let tx = Arc::new(tx);
    let player_map = Arc::new(Mutex::new(None));

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            let api_state = ApiState { tx, player_map };
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
