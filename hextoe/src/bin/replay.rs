//! hextoe-replay — Game replay analyser with per-move value-gap ratings.
//!
//! Search uses heuristic hybrid MCTS (`TacticalRollout` in `hextoe::mcts`): compound-threat
//! priors, parallel root search, tactical-then-random rollouts. Move ranking follows **visit
//! counts**; displayed “value” is a backed-up index in 0–1, not calibrated win probability.
//!
//! Usage:
//!   cargo run --release --bin hextoe-replay -- <game-chunk.json> [game_index]
//!
//! Controls:
//!   ←/→ arrow keys or buttons — step through moves
//!   Click a move in the list  — jump to that move
//!   Scroll/pinch              — zoom the board
//!   Export JSON… / Copy JSON  — dump current board + legal moves + heuristics + `from_cells`
//!                                 snippet (file is written to the process working directory)

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, RichText, Stroke};
use hextoe::game::{opp_straight_extension_blocks, winning_line, GameState, Player, Pos};
use hextoe::mcts::{naive_best_move, Mcts, TacticalRollout};
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

// ── JSON parsing ──────────────────────────────────────────────────────────────

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
struct PlayerJson {
    #[serde(rename = "playerId")]
    player_id: String,
    #[serde(rename = "displayName")]
    display_name: String,
}

#[derive(Deserialize)]
struct GameResultJson {
    #[serde(rename = "winningPlayerId")]
    winning_player_id: Option<String>,
    reason: String,
}

#[derive(Deserialize)]
struct GameJson {
    id: String,
    players: Vec<PlayerJson>,
    #[serde(rename = "gameResult")]
    game_result: GameResultJson,
    moves: Vec<MoveJson>,
}

#[derive(Deserialize)]
struct GamesFile {
    games: Vec<GameJson>,
}

// ── Replay data ───────────────────────────────────────────────────────────────

/// One half-move in a replay.
struct ReplayStep {
    /// Board state BEFORE this move.
    state: GameState,
    /// The move that was played.
    pos: Pos,
    /// Display name of the player who played.
    player_name: String,
    /// The Player enum (X or O).
    player: Player,
}

struct ReplayGame {
    id: String,
    /// player_name[0] = X, player_name[1] = O.
    player_names: [String; 2],
    winner_name: Option<String>,
    result_reason: String,
    steps: Vec<ReplayStep>,
}

fn parse_games(path: &str) -> Result<Vec<ReplayGame>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let file: GamesFile = serde_json::from_str(&content)?;
    let mut games = Vec::new();
    for g in file.games {
        let Some(game) = parse_one(&g) else { continue };
        games.push(game);
    }
    Ok(games)
}

fn parse_one(g: &GameJson) -> Option<ReplayGame> {
    if g.moves.is_empty() { return None; }
    if !matches!(g.game_result.reason.as_str(), "six-in-a-row" | "surrender") {
        return None;
    }
    let mut moves = g.moves.iter().collect::<Vec<_>>();
    moves.sort_by_key(|m| m.move_number);

    let first_id = moves[0].player_id.as_str();
    let name_of = |id: &str| -> String {
        g.players.iter()
            .find(|p| p.player_id == id)
            .map(|p| p.display_name.clone())
            .unwrap_or_else(|| id.to_string())
    };

    let x_name = name_of(first_id);
    let o_id = moves.iter()
        .find(|m| m.player_id != first_id)
        .map(|m| m.player_id.as_str())
        .unwrap_or("");
    let o_name = name_of(o_id);
    let winner_name = g.game_result.winning_player_id.as_deref().map(name_of);

    let mut state = GameState::new();
    let mut steps = Vec::new();
    for m in &moves {
        if state.is_terminal() { break; }
        let pos = (m.x, m.y);
        let player = state.current_player();
        let snapshot = state.clone();
        if !state.place(pos) { continue; }
        let player_name = if m.player_id == first_id { x_name.clone() } else { o_name.clone() };
        steps.push(ReplayStep { state: snapshot, pos, player_name, player });
    }
    if steps.is_empty() { return None; }

    Some(ReplayGame {
        id: g.id.clone(),
        player_names: [x_name, o_name],
        winner_name,
        result_reason: g.game_result.reason.clone(),
        steps,
    })
}

// ── Blunder analysis ──────────────────────────────────────────────────────────

/// MCTS analysis result for one board position.
struct Analysis {
    iters: u32,
    /// All root children sorted by descending score.
    all_moves: Vec<(Pos, f32, u32, f32)>,
}

impl Analysis {
    /// Mean backed-up value for the visit leader, mapped to [0, 1] (not P(win)).
    fn visit_leader_value_display(&self) -> f32 {
        self.all_moves.first().map(|m| m.1).unwrap_or(0.5)
    }

    fn value_display_for(&self, pos: Pos) -> Option<f32> {
        self.all_moves.iter().find(|m| m.0 == pos).map(|m| m.1)
    }

    /// How far `actual` trails the visit leader on the same 0–1 value index (not % win).
    fn value_gap_vs_leader(&self, actual: Pos) -> Option<f32> {
        let best = self.visit_leader_value_display();
        let actual_v = self.value_display_for(actual)?;
        Some((best - actual_v).max(0.0))
    }

    fn rank_of(&self, pos: Pos) -> Option<usize> {
        self.all_moves.iter().position(|m| m.0 == pos)
    }

    fn visit_leader_share(&self) -> Option<f32> {
        self.all_moves.first().map(|m| m.3)
    }
}

fn value_gap_color(gap: f32) -> Color32 {
    if gap < 0.02 { Color32::from_rgb(70, 200, 90) }       // matches visit leader
    else if gap < 0.05 { Color32::from_rgb(220, 210, 50) } // small gap
    else if gap < 0.15 { Color32::from_rgb(230, 140, 40) } // large gap
    else { Color32::from_rgb(210, 50, 50) }                 // very large gap
}

fn value_gap_label(gap: f32) -> &'static str {
    if gap < 0.02 { "Matches leader" }
    else if gap < 0.05 { "Small gap" }
    else if gap < 0.15 { "Large gap" }
    else { "Very large gap" }
}

// ── Hex geometry (copied from main.rs) ───────────────────────────────────────

fn hex_to_pixel(q: i32, r: i32, size: f32, origin: Pos2) -> Pos2 {
    Pos2::new(
        origin.x + size * (3f32.sqrt() * q as f32 + 3f32.sqrt() / 2.0 * r as f32),
        origin.y + size * (1.5 * r as f32),
    )
}

fn pixel_to_hex(p: Pos2, size: f32, origin: Pos2) -> Pos {
    let x = p.x - origin.x;
    let y = p.y - origin.y;
    hex_round(
        (3f32.sqrt() / 3.0 * x - 1.0 / 3.0 * y) / size,
        (2.0 / 3.0 * y) / size,
    )
}

fn hex_round(fq: f32, fr: f32) -> Pos {
    let fs = -fq - fr;
    let rq = fq.round(); let rr = fr.round(); let rs = fs.round();
    let dq = (rq - fq).abs(); let dr = (rr - fr).abs(); let ds = (rs - fs).abs();
    if dq > dr && dq > ds { ((-rr - rs) as i32, rr as i32) }
    else if dr > ds { (rq as i32, (-rq - rs) as i32) }
    else { (rq as i32, rr as i32) }
}

fn hex_corners(centre: Pos2, size: f32) -> [Pos2; 6] {
    std::array::from_fn(|i| {
        let angle = std::f32::consts::FRAC_PI_6 + std::f32::consts::FRAC_PI_3 * i as f32;
        Pos2::new(centre.x + size * angle.cos(), centre.y + size * angle.sin())
    })
}

fn player_color(p: Player) -> Color32 {
    match p { Player::X => Color32::from_rgb(230, 120, 30), Player::O => Color32::from_rgb(70, 140, 220) }
}
fn player_dark(p: Player) -> Color32 {
    match p { Player::X => Color32::from_rgb(170, 85, 15), Player::O => Color32::from_rgb(45, 95, 165) }
}

// ── App ───────────────────────────────────────────────────────────────────────

/// Iterations per analysis chunk (parallel MCTS makes larger batches affordable).
const BATCH: u32 = 800;

struct ReplayApp {
    games: Vec<ReplayGame>,
    game_idx: usize,
    /// Current move index (0 = board before any moves; step N shows state after N moves).
    move_idx: usize,
    /// Path passed on the command line (for export metadata).
    source_json_path: String,

    analysis: Option<Analysis>,
    cancel: Arc<AtomicBool>,
    result_rx: Option<Receiver<Analysis>>,

    pan_offset: egui::Vec2,
    hex_size: f32,
    /// Last export / copy status for the user.
    export_status: Option<String>,
}

impl ReplayApp {
    fn new(games: Vec<ReplayGame>, start_game: usize, source_json_path: String) -> Self {
        let mut app = ReplayApp {
            games,
            game_idx: start_game,
            move_idx: 0,
            source_json_path,
            analysis: None,
            cancel: Arc::new(AtomicBool::new(false)),
            result_rx: None,
            pan_offset: egui::Vec2::ZERO,
            hex_size: 28.0,
            export_status: None,
        };
        app.restart_analysis();
        app
    }

    fn game(&self) -> &ReplayGame { &self.games[self.game_idx] }

    /// Current board state (state BEFORE move_idx was played, or final state if at end).
    fn current_state(&self) -> &GameState {
        let g = self.game();
        if self.move_idx < g.steps.len() {
            &g.steps[self.move_idx].state
        } else {
            // Past end: show final board (state after last move).
            // Reconstruct: state of last step + the last move applied.
            // We don't store final state explicitly, so just return the last step's state.
            // (The board will show pieces up to but not including the last move.)
            // Better: we add a terminal state entry. For now return last step state.
            &g.steps[g.steps.len() - 1].state
        }
    }

    fn total_moves(&self) -> usize { self.game().steps.len() }

    fn restart_analysis(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.analysis = None;
        self.result_rx = None;

        let state = self.current_state().clone();
        if state.is_terminal() { return; }

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let mut mcts = Mcts::new(state);
            let rollout = TacticalRollout;

            loop {
                if cancel.load(Ordering::Relaxed) { break; }
                mcts.search_iters(BATCH, &mut rng, &rollout);
                let iters = mcts.total_visits();
                // Return ALL root children (not just top 3) so we can look up any move.
                let all_moves = mcts.best_moves(usize::MAX);
                let a = Analysis { iters, all_moves };
                if tx.send(a).is_err() { break; }
            }
        });

        self.result_rx = Some(rx);
    }

    fn go_to_move(&mut self, idx: usize) {
        let clamped = idx.min(self.total_moves().saturating_sub(1));
        if clamped != self.move_idx {
            self.move_idx = clamped;
            self.restart_analysis();
        }
    }

    fn go_to_game(&mut self, idx: usize) {
        self.game_idx = idx;
        self.move_idx = 0;
        self.pan_offset = egui::Vec2::ZERO;
        self.restart_analysis();
    }

    fn step_forward(&mut self) {
        let next = self.move_idx + 1;
        if next < self.total_moves() {
            self.move_idx = next;
            self.restart_analysis();
        }
    }

    fn step_back(&mut self) {
        if self.move_idx > 0 {
            self.move_idx -= 1;
            self.restart_analysis();
        }
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

impl eframe::App for ReplayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain analysis results.
        if let Some(rx) = &self.result_rx {
            let mut latest = None;
            while let Ok(a) = rx.try_recv() { latest = Some(a); }
            if let Some(a) = latest { self.analysis = Some(a); }
            ctx.request_repaint();
        }

        // Keyboard nav — apply before building UI.
        let fwd = ctx.input(|i| i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowDown));
        let bck = ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)  || i.key_pressed(egui::Key::ArrowUp));
        if fwd { self.step_forward(); }
        if bck { self.step_back(); }

        // ── Extract all display data as owned values so closures don't hold ──
        // a reference into self while we also need &mut self for pan/zoom.
        let move_idx   = self.move_idx;
        let game_idx   = self.game_idx;
        let _num_games = self.games.len();

        // Game-level data (cloned).
        let x_name      = self.games[game_idx].player_names[0].clone();
        let o_name      = self.games[game_idx].player_names[1].clone();
        let winner_name = self.games[game_idx].winner_name.clone();
        let result_reason = self.games[game_idx].result_reason.clone();
        let total       = self.games[game_idx].steps.len();

        // Current step.
        let step_pos: Option<Pos>    = self.games[game_idx].steps.get(move_idx).map(|s| s.pos);
        let step_player: Option<Player> = self.games[game_idx].steps.get(move_idx).map(|s| s.player);
        let step_name: Option<String>   = self.games[game_idx].steps.get(move_idx).map(|s| s.player_name.clone());

        // Analysis snapshot (cheap clone of vecs of small structs).
        let analysis_snapshot: Option<Analysis> = self.analysis.as_ref().map(|a| Analysis {
            iters: a.iters,
            all_moves: a.all_moves.clone(),
        });

        let best_pos: Option<Pos> = analysis_snapshot.as_ref()
            .and_then(|a| a.all_moves.first().map(|m| m.0));

        // Game-list labels (cloned once).
        let game_labels: Vec<String> = self.games.iter().enumerate().map(|(i, g)| {
            format!("#{} {} vs {}", i + 1, g.player_names[0], g.player_names[1])
        }).collect();

        // Timeline steps (cloned for the bottom panel).
        let timeline: Vec<(Player, Pos)> = self.games[game_idx].steps.iter()
            .map(|s| (s.player, s.pos)).collect();

        // Board state and move-number map (cloned).
        let board_state: GameState = self.games[game_idx].steps
            .get(move_idx)
            .map(|s| s.state.clone())
            .unwrap_or_else(|| {
                // Past end: reconstruct by replaying all moves.
                let mut st = GameState::new();
                for s in &self.games[game_idx].steps { st.place(s.pos); }
                st
            });

        // Final board state for win-line.
        let final_state: GameState = {
            let mut st = GameState::new();
            for s in &self.games[game_idx].steps { st.place(s.pos); }
            st
        };
        let last_pos = self.games[game_idx].steps.last().map(|s| s.pos);
        let win_cells: std::collections::HashSet<Pos> = last_pos
            .and_then(|lp| final_state.winner.map(|w| winning_line(&final_state.board, lp, w).into_iter().collect()))
            .unwrap_or_default();

        let mut move_numbers: std::collections::HashMap<Pos, usize> = Default::default();
        for (i, s) in self.games[game_idx].steps.iter().enumerate() {
            if i < move_idx { move_numbers.insert(s.pos, i + 1); }
        }

        // Blunder fill for actual move.
        let actual_blunder_fill: Option<Color32> = step_pos
            .and_then(|p| analysis_snapshot.as_ref()?.value_gap_vs_leader(p))
            .map(value_gap_color);

        // ── Deferred actions ──────────────────────────────────────────────
        let mut game_select:  Option<usize> = None;
        let mut move_select:  Option<usize> = None;
        let mut nav_fwd = false;
        let mut nav_bck = false;
        let mut pan_delta   = egui::Vec2::ZERO;
        let mut zoom_action: Option<(Pos2, f32)> = None;
        let mut scroll_delta = egui::Vec2::ZERO;
        let mut list_hover_pos: Option<Pos> = None;
        let export_note = self.export_status.clone();

        // ── Left panel: game list ──────────────────────────────────────────
        egui::SidePanel::left("games")
            .min_width(160.0).max_width(200.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(RichText::new("Games").strong());
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, label) in game_labels.iter().enumerate() {
                        if ui.selectable_label(i == game_idx, label).clicked() {
                            game_select = Some(i);
                        }
                    }
                });
            });

        // ── Right panel: analysis ──────────────────────────────────────────
        let mut export_to_file = false;
        let mut copy_json = false;

        egui::SidePanel::right("analysis")
            .exact_width(280.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.colored_label(player_color(Player::X), RichText::new(&x_name).strong());
                    ui.label(" vs ");
                    ui.colored_label(player_color(Player::O), RichText::new(&o_name).strong());
                });
                if let Some(ref w) = winner_name {
                    ui.label(format!("Winner: {} ({})", w, result_reason));
                }
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("◀").clicked() { nav_bck = true; }
                    ui.label(format!("Move {}/{}", move_idx + 1, total));
                    if ui.button("▶").clicked() { nav_fwd = true; }
                });
                ui.separator();

                if let (Some(name), Some(player), Some(pos)) = (&step_name, step_player, step_pos) {
                    ui.colored_label(player_color(player), RichText::new(name).strong());
                    ui.label(format!("Played: ({}, {})", pos.0, pos.1));
                }
                ui.separator();

                match &analysis_snapshot {
                    None => { ui.label("Analysing…"); ui.spinner(); }
                    Some(a) => {
                        ui.label(format!("MCTS simulations: {}", fmt_iters(a.iters)));
                        ui.label(
                            RichText::new(
                                "Heuristic hybrid · parallel · rank by visits · 0–1 column = mean backup (not P(win))",
                            )
                            .small()
                            .weak(),
                        );
                        if let Some(apos) = step_pos {
                            if let Some(gap) = a.value_gap_vs_leader(apos) {
                                egui::Frame::none()
                                    .fill(value_gap_color(gap))
                                    .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                                    .rounding(egui::Rounding::same(6.0))
                                    .show(ui, |ui| {
                                        ui.label(RichText::new(value_gap_label(gap))
                                            .strong().color(Color32::WHITE).size(18.0));
                                        ui.label(RichText::new(format!(
                                            "Value index gap vs visit leader: {:.2} (not % win chance)",
                                            gap
                                        ))
                                            .small()
                                            .color(Color32::WHITE));
                                    });
                                if let Some(r) = a.rank_of(apos) {
                                    ui.label(format!("Visit rank: #{} (among explored root moves)", r + 1));
                                }
                            } else {
                                ui.label("(move not yet explored)");
                            }
                            if let Some(bpos) = best_pos {
                                ui.add_space(4.0);
                                if bpos == apos {
                                    ui.colored_label(Color32::from_rgb(60, 200, 80), "✓ Same as visit leader");
                                } else {
                                    let v = a.visit_leader_value_display();
                                    ui.colored_label(Color32::from_rgb(60, 200, 80),
                                        format!("Visit leader: ({},{})  ·  mean value index ≈ {:.2}",
                                            bpos.0, bpos.1, v));
                                    if let Some(sh) = a.visit_leader_share() {
                                        ui.label(RichText::new(format!("Leader visit share: {:.0}%", sh * 100.0)).small().weak());
                                    }
                                }
                            }
                        }
                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(RichText::new("Top moves (by visits)").strong());
                        for (rank, (pos, score, visits, p_share)) in a.all_moves.iter().take(5).enumerate() {
                            let is_actual = Some(*pos) == step_pos;
                            let col = if is_actual { step_player.map(player_color).unwrap_or(Color32::GRAY) }
                                else if rank == 0 { Color32::from_rgb(60, 200, 80) }
                                else { Color32::GRAY };
                            let text = format!(
                                "{} #{} ({},{})  {} visits ({:.0}%)  ·  idx {:.2}",
                                if is_actual { "●" } else { "○" },
                                rank + 1, pos.0, pos.1,
                                fmt_iters(*visits), p_share * 100.0, score
                            );
                            let resp = ui.add(
                                egui::Label::new(RichText::new(text).color(col))
                                    .sense(egui::Sense::hover()),
                            );
                            if resp.hovered() {
                                list_hover_pos = Some(*pos);
                            }
                        }
                    }
                }
                ui.separator();
                ui.label(RichText::new("Controls").strong());
                ui.label("← → arrow keys or buttons");
                ui.label("Click timeline to jump");
                ui.separator();
                ui.label(RichText::new("Export").strong());
                ui.horizontal(|ui| {
                    if ui
                        .button("Export JSON…")
                        .on_hover_text("Write hextoe_replay_export_gameN_moveM.json in the current working directory")
                        .clicked()
                    {
                        export_to_file = true;
                    }
                    if ui
                        .button("Copy JSON")
                        .on_hover_text("Full snapshot for tests, issues, or sharing")
                        .clicked()
                    {
                        copy_json = true;
                    }
                });
                if let Some(ref n) = export_note {
                    ui.label(RichText::new(n).small().weak());
                }
            });

        // ── Bottom panel: move timeline ────────────────────────────────────
        egui::TopBottomPanel::bottom("moves")
            .min_height(40.0).max_height(60.0)
            .show(ctx, |ui| {
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (i, &(player, pos)) in timeline.iter().enumerate() {
                            let label = format!("{}. ({},{})", i + 1, pos.0, pos.1);
                            let rt = RichText::new(&label).color(player_color(player));
                            let rt = if i == move_idx { rt.strong() } else { rt };
                            if ui.selectable_label(i == move_idx, rt).clicked() {
                                move_select = Some(i);
                            }
                        }
                    });
                });
            });

        // ── Central board panel ────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            if response.dragged() { pan_delta = response.drag_delta(); }
            if response.hovered() {
                let zoom = ctx.input(|i| i.zoom_delta());
                if zoom != 1.0 {
                    let ptr = ctx.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                    zoom_action = Some((ptr, zoom));
                }
                let scroll = ctx.input(|i| i.smooth_scroll_delta);
                if scroll != egui::Vec2::ZERO { scroll_delta = scroll; }
            }

            let hs = self.hex_size;
            let origin = rect.center() + self.pan_offset;

            let mut cells: std::collections::HashSet<Pos> = board_state.board.keys().copied().collect();
            if cells.is_empty() {
                for dq in -3i32..=3 {
                    for dr in (-3i32).max(-dq-3)..=3i32.min(-dq+3) { cells.insert((dq, dr)); }
                }
            }
            for &p in &board_state.candidates { cells.insert(p); }
            if let Some(p) = best_pos { cells.insert(p); }
            if let Some(p) = step_pos  { cells.insert(p); }
            if let Some(p) = list_hover_pos { cells.insert(p); }

            let painter = ui.painter_at(rect);
            let expanded = rect.expand(hs * 3.0);

            for &pos in &cells {
                let centre = hex_to_pixel(pos.0, pos.1, hs, origin);
                if !expanded.contains(centre) { continue; }

                let is_best   = best_pos  == Some(pos);
                let is_actual = step_pos  == Some(pos);
                let is_list_hover = list_hover_pos == Some(pos);
                let corners: Vec<Pos2> = hex_corners(centre, hs - 1.5).into();

                let fill = if let Some(&player) = board_state.board.get(&pos) {
                    if win_cells.contains(&pos) {
                        match player {
                            Player::X => Color32::from_rgb(255, 175, 80),
                            Player::O => Color32::from_rgb(110, 195, 255),
                        }
                    } else { player_dark(player) }
                } else if is_actual && is_best { Color32::from_rgb(45, 185, 45) }
                  else if is_best  { Color32::from_rgb(45, 185, 45) }
                  else if is_actual { actual_blunder_fill.unwrap_or(Color32::from_gray(80)) }
                  else { Color32::from_gray(52) };

                let stroke_w = if is_list_hover {
                    3.0
                } else if is_actual || is_best || win_cells.contains(&pos) {
                    2.5
                } else {
                    1.0
                };
                let stroke_col = if is_list_hover {
                    Color32::from_rgb(255, 220, 90)
                } else if is_actual && is_best {
                    Color32::WHITE
                } else if is_best {
                    Color32::from_rgb(100, 255, 100)
                } else if is_actual {
                    Color32::WHITE
                } else if win_cells.contains(&pos) {
                    Color32::YELLOW
                } else {
                    Color32::from_gray(35)
                };

                painter.add(egui::Shape::convex_polygon(corners, fill, Stroke::new(stroke_w, stroke_col)));

                let font = FontId::proportional((hs * 0.32).clamp(8.0, 16.0));
                if let Some(&mn) = move_numbers.get(&pos) {
                    painter.text(centre, Align2::CENTER_CENTER, mn.to_string(), font, Color32::WHITE);
                } else if is_best && !is_actual {
                    painter.text(centre, Align2::CENTER_CENTER, "★", font, Color32::WHITE);
                } else if is_actual {
                    if let Some(bp) = analysis_snapshot.as_ref().and_then(|a| a.value_gap_vs_leader(pos)) {
                        painter.text(centre, Align2::CENTER_CENTER,
                            if bp < 0.02 { "✓" } else if bp < 0.05 { "?" } else { "✗" },
                            font, Color32::WHITE);
                    }
                }
            }

            // On-board nav buttons.
            let nav_l = egui::Rect::from_min_size(
                rect.min + egui::vec2(8.0, rect.height() / 2.0 - 20.0), egui::vec2(60.0, 40.0));
            let nav_r = egui::Rect::from_min_size(
                egui::pos2(rect.max.x - 68.0, rect.min.y + rect.height() / 2.0 - 20.0), egui::vec2(60.0, 40.0));
            if ui.put(nav_l, egui::Button::new(RichText::new("◀").size(22.0))).clicked() { nav_bck = true; }
            if ui.put(nav_r, egui::Button::new(RichText::new("▶").size(22.0))).clicked() { nav_fwd = true; }
        });

        // ── Apply deferred mutations ───────────────────────────────────────
        self.pan_offset += pan_delta + scroll_delta;
        if let Some((ptr, zoom)) = zoom_action {
            // Zoom toward pointer (requires a dummy rect centre; use current pan).
            let old = self.hex_size;
            let new = (old * zoom).clamp(8.0, 120.0);
            let z = new / old;
            // We don't have the rect centre here, but egui centres the board on screen centre.
            // Use pointer relative to origin approx.
            self.pan_offset = ptr.to_vec2() * (1.0 - z) + z * self.pan_offset;
            self.hex_size = new;
        }
        if nav_fwd { self.step_forward(); }
        if nav_bck { self.step_back(); }
        if let Some(i) = move_select { self.go_to_move(i); }
        if let Some(i) = game_select { self.go_to_game(i); }

        if export_to_file || copy_json {
            let payload = build_export_payload(
                &self.source_json_path,
                self.game(),
                game_idx,
                move_idx,
                &board_state,
                analysis_snapshot.as_ref(),
            );
            match serde_json::to_string_pretty(&payload) {
                Ok(text) => {
                    let mut parts: Vec<String> = Vec::new();
                    if export_to_file {
                        let fname =
                            format!("hextoe_replay_export_game{}_move{}.json", game_idx + 1, move_idx);
                        match std::fs::write(&fname, &text) {
                            Ok(()) => parts.push(format!("Saved {fname} (process cwd).")),
                            Err(e) => parts.push(format!("Could not write file: {e}.")),
                        }
                    }
                    if copy_json {
                        ctx.copy_text(text);
                        parts.push("Copied JSON to clipboard.".into());
                    }
                    self.export_status = Some(parts.join(" "));
                }
                Err(e) => self.export_status = Some(format!("Serialize failed: {e}")),
            }
        }
    }
}

fn fmt_iters(n: u32) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}k", n as f32 / 1_000.0) }
    else { format!("{n}") }
}

fn player_label(p: Player) -> &'static str {
    match p {
        Player::X => "X",
        Player::O => "O",
    }
}

/// JSON + Rust snippet for regression tests and offline debugging.
fn build_export_payload(
    source_json_path: &str,
    game: &ReplayGame,
    game_idx: usize,
    move_idx: usize,
    state: &GameState,
    analysis: Option<&Analysis>,
) -> serde_json::Value {
    let mut cell_entries: Vec<(Pos, Player)> =
        state.board.iter().map(|(p, pl)| (*p, *pl)).collect();
    cell_entries.sort_by_key(|(p, _)| (p.0, p.1));

    let cells: Vec<serde_json::Value> = cell_entries
        .iter()
        .map(|(p, pl)| {
            json!({
                "x": p.0,
                "y": p.1,
                "player": player_label(*pl),
            })
        })
        .collect();

    let mut legal: Vec<Pos> = state.legal_actions();
    legal.sort_by_key(|p| (p.0, p.1));
    let legal_json: Vec<serde_json::Value> = legal.iter().map(|p| json!([p.0, p.1])).collect();

    let opp = state.current_player().other();
    let mut straight: Vec<Pos> = opp_straight_extension_blocks(&state.board, opp).into_iter().collect();
    straight.sort_by_key(|p| (p.0, p.1));
    let straight_json: Vec<serde_json::Value> = straight.iter().map(|p| json!([p.0, p.1])).collect();

    let rust_cell_lines: Vec<String> = cell_entries
        .iter()
        .map(|(p, pl)| {
            format!(
                "    (({}, {}), Player::{}),",
                p.0,
                p.1,
                player_label(*pl)
            )
        })
        .collect();
    let cp = state.current_player();
    let rust_snippet = format!(
        concat!(
            "// Paste into a `#[test]` or bench (hextoe::game).\n",
            "let cells: &[(Pos, Player)] = &[\n",
            "{}\n",
            "];\n",
            "let total_moves = {}u32;\n",
            "let state = GameState::from_cells(cells, total_moves);\n",
            "assert_eq!(state.current_player(), Player::{});\n"
        ),
        rust_cell_lines.join("\n"),
        state.total_moves,
        player_label(cp),
    );

    let mcts_json = analysis.map(|a| {
        json!({
            "simulations": a.iters,
            "top_by_visits": a.all_moves.iter().take(24).map(|(p, s, v, ps)| {
                json!({
                    "pos": [p.0, p.1],
                    "mean_value_index": s,
                    "visits": v,
                    "visit_share": ps,
                })
            }).collect::<Vec<_>>(),
        })
    });

    json!({
        "meta": {
            "source_json_file": source_json_path,
            "game_list_index_0": game_idx,
            "game_id": game.id,
            "replay_move_index_0": move_idx,
            "note": "Board is the position BEFORE the move at replay_move_index (same as analyser board).",
            "player_x_display": game.player_names[0],
            "player_o_display": game.player_names[1],
        },
        "game_state": {
            "total_moves": state.total_moves,
            "current_player": player_label(cp),
            "turn_label": state.turn_label(),
            "is_terminal": state.is_terminal(),
            "winner": state.winner.map(|w| player_label(w)),
            "legal_action_count": legal.len(),
        },
        "cells": cells,
        "legal_actions": legal_json,
        "debug_heuristics": {
            "opponent_to_block": player_label(opp),
            "opp_straight_extension_blocks": straight_json,
            "naive_best_move": naive_best_move(state).map(|p| json!([p.0, p.1])),
        },
        "rust_from_cells_snippet": rust_snippet,
        "mcts_snapshot": mcts_json,
    })
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: hextoe-replay <game-chunk.json> [game_index]");
        std::process::exit(1);
    }
    let path = &args[1];
    let start_game: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let games = parse_games(path).unwrap_or_else(|e| {
        eprintln!("Failed to load {path}: {e}");
        std::process::exit(1);
    });

    if games.is_empty() {
        eprintln!("No valid games found in {path}");
        std::process::exit(1);
    }

    println!("Loaded {} games from {path}", games.len());

    let start = start_game.min(games.len() - 1);
    let source_path = args[1].clone();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 780.0])
            .with_title("Hextoe Replay Analyser"),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Hextoe Replay",
        options,
        Box::new(move |_cc| Ok(Box::new(ReplayApp::new(games, start, source_path)))),
    )
}
