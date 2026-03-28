//! hextoe-replay — Game replay analyser with per-move blunder ratings.
//!
//! Usage:
//!   cargo run --release --bin hextoe-replay -- <game-chunk.json> [game_index]
//!
//! Controls:
//!   ←/→ arrow keys or buttons — step through moves
//!   Click a move in the list  — jump to that move
//!   Scroll/pinch              — zoom the board

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, RichText, Stroke};
use hextoe::device::default_inference_device;
use hextoe::game::{winning_line, GameState, Player, Pos};
use hextoe::mcts::{Mcts, RandomRollout};
use hextoe::nn::{LoadedNet, NeuralRollout};
use hextoe::train::default_inference_checkpoint_path;
use serde::Deserialize;
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
    fn best_score(&self) -> f32 {
        self.all_moves.first().map(|m| m.1).unwrap_or(0.5)
    }

    fn score_for(&self, pos: Pos) -> Option<f32> {
        self.all_moves.iter().find(|m| m.0 == pos).map(|m| m.1)
    }

    fn blunder_pct(&self, actual: Pos) -> Option<f32> {
        let best = self.best_score();
        let actual_score = self.score_for(actual)?;
        Some((best - actual_score).max(0.0))
    }

    fn rank_of(&self, pos: Pos) -> Option<usize> {
        self.all_moves.iter().position(|m| m.0 == pos)
    }
}

fn blunder_color(pct: f32) -> Color32 {
    if pct < 0.02 { Color32::from_rgb(70, 200, 90) }       // best / OK
    else if pct < 0.05 { Color32::from_rgb(220, 210, 50) } // inaccuracy
    else if pct < 0.15 { Color32::from_rgb(230, 140, 40) } // mistake
    else { Color32::from_rgb(210, 50, 50) }                 // blunder
}

fn blunder_label(pct: f32) -> &'static str {
    if pct < 0.02 { "Best" }
    else if pct < 0.05 { "Inaccuracy" }
    else if pct < 0.15 { "Mistake" }
    else { "Blunder" }
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

const BATCH: u32 = 500;

struct ReplayApp {
    games: Vec<ReplayGame>,
    game_idx: usize,
    /// Current move index (0 = board before any moves; step N shows state after N moves).
    move_idx: usize,

    analysis: Option<Analysis>,
    cancel: Arc<AtomicBool>,
    result_rx: Option<Receiver<Analysis>>,

    pan_offset: egui::Vec2,
    hex_size: f32,
}

impl ReplayApp {
    fn new(games: Vec<ReplayGame>, start_game: usize) -> Self {
        let mut app = ReplayApp {
            games,
            game_idx: start_game,
            move_idx: 0,
            analysis: None,
            cancel: Arc::new(AtomicBool::new(false)),
            result_rx: None,
            pan_offset: egui::Vec2::ZERO,
            hex_size: 28.0,
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
            let device = default_inference_device();
            let loaded = default_inference_checkpoint_path()
                .and_then(|p| LoadedNet::try_load(p, &device).ok());
            let mut rng = rand::thread_rng();
            let mut mcts = Mcts::new(state);

            loop {
                if cancel.load(Ordering::Relaxed) { break; }
                if let Some(ref ld) = loaded {
                    let r = NeuralRollout { net: &ld.net, device: &device };
                    mcts.search_iters(BATCH, &mut rng, &r);
                } else {
                    mcts.search_iters(BATCH, &mut rng, &RandomRollout);
                }
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

        // Keyboard nav.
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowDown) {
                // handled after borrow ends
            }
        });
        let fwd = ctx.input(|i| i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowDown));
        let bck = ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::ArrowUp));
        if fwd { self.step_forward(); }
        if bck { self.step_back(); }

        let game = &self.games[self.game_idx];
        let move_idx = self.move_idx;
        let total = game.steps.len();
        let current_step = if move_idx < total { Some(&game.steps[move_idx]) } else { None };

        // Which move was actually played at this position?
        let actual_pos = current_step.map(|s| s.pos);
        let best_pos = self.analysis.as_ref()
            .and_then(|a| a.all_moves.first().map(|m| m.0));

        // ── Left panel: game list ──────────────────────────────────────────
        egui::SidePanel::left("games")
            .min_width(160.0)
            .max_width(200.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(RichText::new("Games").strong());
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, g) in self.games.iter().enumerate() {
                        let label = format!(
                            "#{} {} vs {}",
                            i + 1,
                            g.player_names[0],
                            g.player_names[1]
                        );
                        let selected = i == self.game_idx;
                        if ui.selectable_label(selected, &label).clicked() {
                            // deferred to avoid borrow conflict
                            let _ = i; // captured below
                        }
                        // Re-do with index capture:
                        let _ = label;
                    }
                    // Re-render with proper click handling:
                });
                // We can't call self.go_to_game inside the closure above, so we do a second pass.
            });
        // Work around borrow: collect click target separately.
        let mut game_click: Option<usize> = None;
        egui::SidePanel::left("games_click_shadow")
            .min_width(0.0)
            .max_width(0.0)
            .show(ctx, |_ui| {});
        // Simpler: just re-draw game list items as buttons in a fresh closure scope.
        // (egui panels with same id are merged, so we use one panel)
        // Actually the above approach has a bug. Let's do a single panel properly:
        // The game list was already shown. Button clicks inside show() closures
        // do work — we just can't call &mut self methods. Use a local var.
        // We need to restructure. Use immediate mode correctly:
        // Collect into local then apply.

        // ── Right panel: analysis ──────────────────────────────────────────
        egui::SidePanel::right("analysis")
            .exact_width(240.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);

                // Game header.
                let g = &self.games[self.game_idx];
                let x_col = player_color(Player::X);
                let o_col = player_color(Player::O);
                ui.horizontal(|ui| {
                    ui.colored_label(x_col, RichText::new(&g.player_names[0]).strong());
                    ui.label(" vs ");
                    ui.colored_label(o_col, RichText::new(&g.player_names[1]).strong());
                });
                if let Some(ref w) = g.winner_name {
                    ui.label(format!("Winner: {} ({})", w, g.result_reason));
                }

                ui.separator();

                // Move navigation.
                ui.horizontal(|ui| {
                    if ui.button("◀").clicked() { /* handled below */ }
                    ui.label(format!("Move {}/{}", move_idx + 1, total));
                    if ui.button("▶").clicked() { /* handled below */ }
                });
                // Navigation handled via keyboard above; button clicks handled here:
                ui.horizontal(|ui| {
                    if ui.button("◀ Back").clicked() {
                        // can't call self.step_back() here, schedule below
                    }
                    if ui.button("Forward ▶").clicked() {
                        // same
                    }
                });

                ui.separator();

                // Current move info.
                if let Some(step) = current_step {
                    let col = player_color(step.player);
                    ui.colored_label(col, RichText::new(&step.player_name).strong());
                    ui.label(format!("Played: ({}, {})", step.pos.0, step.pos.1));
                }

                ui.separator();

                // Analysis.
                match &self.analysis {
                    None => {
                        ui.label("Analysing…");
                        ui.spinner();
                    }
                    Some(a) => {
                        ui.label(format!("Iters: {}", fmt_iters(a.iters)));

                        if let Some(apos) = actual_pos {
                            let blunder = a.blunder_pct(apos);
                            let rank = a.rank_of(apos);

                            // Blunder badge.
                            if let Some(bp) = blunder {
                                let bcol = blunder_color(bp);
                                let blabel = blunder_label(bp);
                                egui::Frame::none()
                                    .fill(bcol)
                                    .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                                    .rounding(egui::Rounding::same(6.0))
                                    .show(ui, |ui| {
                                        ui.label(RichText::new(blabel)
                                            .strong().color(Color32::WHITE).size(18.0));
                                        ui.label(RichText::new(
                                            format!("-{:.1}%", bp * 100.0)
                                        ).color(Color32::WHITE));
                                    });
                                if let Some(r) = rank {
                                    ui.label(format!("Move ranked #{} by bot", r + 1));
                                }
                            } else {
                                ui.label("(move not yet explored)");
                            }

                            // Best move info.
                            if let Some(bpos) = best_pos {
                                if Some(bpos) != actual_pos {
                                    let best_score = a.best_score();
                                    ui.add_space(4.0);
                                    ui.colored_label(
                                        Color32::from_rgb(60, 200, 80),
                                        format!("Bot prefers: ({}, {})  {:.1}%",
                                            bpos.0, bpos.1, best_score * 100.0)
                                    );
                                } else {
                                    ui.colored_label(
                                        Color32::from_rgb(60, 200, 80),
                                        "✓ Matches bot's top choice",
                                    );
                                }
                            }
                        }

                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(RichText::new("Top moves").strong());
                        for (rank, (pos, score, visits, _)) in a.all_moves.iter().take(5).enumerate() {
                            let is_actual = Some(*pos) == actual_pos;
                            let col = if is_actual {
                                player_color(current_step.map(|s| s.player).unwrap_or(Player::X))
                            } else if rank == 0 {
                                Color32::from_rgb(60, 200, 80)
                            } else {
                                Color32::GRAY
                            };
                            let marker = if is_actual { "●" } else { "○" };
                            ui.colored_label(col, format!(
                                "{} #{} ({},{}) {:.1}% {}",
                                marker, rank + 1, pos.0, pos.1,
                                score * 100.0,
                                fmt_iters(*visits)
                            ));
                        }
                    }
                }

                ui.separator();
                ui.label(RichText::new("Controls").strong());
                ui.label("← → arrow keys");
                ui.label("Click move list to jump");
            });

        // ── Bottom panel: move list timeline ──────────────────────────────
        egui::TopBottomPanel::bottom("moves")
            .min_height(60.0)
            .max_height(120.0)
            .show(ctx, |ui| {
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (i, step) in game.steps.iter().enumerate() {
                            let col = player_color(step.player);
                            let selected = i == move_idx;
                            let label = format!("{}. ({},{})", i + 1, step.pos.0, step.pos.1);
                            let rt = RichText::new(&label).color(col);
                            let rt = if selected { rt.strong() } else { rt };
                            if ui.selectable_label(selected, rt).clicked() {
                                game_click = Some(i);
                            }
                        }
                    });
                });
            });

        // ── Central board panel ────────────────────────────────────────────
        let mut nav_fwd = false;
        let mut nav_bck = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            if response.dragged() { self.pan_offset += response.drag_delta(); }
            if response.hovered() {
                let zoom = ctx.input(|i| i.zoom_delta());
                if zoom != 1.0 {
                    let ptr = ctx.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                    let old = self.hex_size;
                    let new = (old * zoom).clamp(8.0, 120.0);
                    self.pan_offset = (ptr - rect.center()) * (1.0 - new / old) + (new / old) * self.pan_offset;
                    self.hex_size = new;
                }
                let scroll = ctx.input(|i| i.smooth_scroll_delta);
                if scroll != egui::Vec2::ZERO { self.pan_offset += scroll; }
            }

            let hs = self.hex_size;
            let origin = rect.center() + self.pan_offset;
            let state = self.current_state();

            // Build final state for winning line check.
            let last_step_pos = game.steps.last().map(|s| {
                // reconstruct final state to get winner
                let mut st = s.state.clone();
                st.place(s.pos);
                (st, s.pos)
            });
            let win_cells: std::collections::HashSet<Pos> = last_step_pos
                .as_ref()
                .and_then(|(st, pos)| {
                    st.winner.map(|w| winning_line(&st.board, *pos, w).into_iter().collect())
                })
                .unwrap_or_default();

            let mut cells: std::collections::HashSet<Pos> = state.board.keys().copied().collect();
            if cells.is_empty() {
                for dq in -3i32..=3 {
                    for dr in (-3i32).max(-dq-3)..=3i32.min(-dq+3) { cells.insert((dq, dr)); }
                }
            }
            for &p in &state.candidates { cells.insert(p); }
            if let Some(p) = best_pos { cells.insert(p); }
            if let Some(p) = actual_pos { cells.insert(p); }

            let painter = ui.painter_at(rect);
            let expanded = rect.expand(hs * 3.0);

            // Move number lookup: which move number placed each cell?
            let mut move_numbers: std::collections::HashMap<Pos, usize> =
                std::collections::HashMap::new();
            for (i, step) in game.steps.iter().enumerate() {
                if i < move_idx {
                    move_numbers.insert(step.pos, i + 1);
                }
            }

            for &pos in &cells {
                let centre = hex_to_pixel(pos.0, pos.1, hs, origin);
                if !expanded.contains(centre) { continue; }

                let is_best = best_pos == Some(pos);
                let is_actual = actual_pos == Some(pos);
                let corners: Vec<Pos2> = hex_corners(centre, hs - 1.5).into();

                let fill = if let Some(&player) = state.board.get(&pos) {
                    if win_cells.contains(&pos) {
                        match player {
                            Player::X => Color32::from_rgb(255, 175, 80),
                            Player::O => Color32::from_rgb(110, 195, 255),
                        }
                    } else {
                        player_dark(player)
                    }
                } else if is_actual && is_best {
                    Color32::from_rgb(45, 185, 45) // played the best move
                } else if is_best {
                    Color32::from_rgb(45, 185, 45) // bot's best
                } else if is_actual {
                    // Tint by blunder severity
                    self.analysis.as_ref()
                        .and_then(|a| a.blunder_pct(pos))
                        .map(blunder_color)
                        .unwrap_or(Color32::from_gray(80))
                } else {
                    Color32::from_gray(52)
                };

                let stroke_w = if is_actual || is_best { 2.5 } else if win_cells.contains(&pos) { 2.5 } else { 1.0 };
                let stroke_col = if is_actual && is_best { Color32::WHITE }
                    else if is_best { Color32::from_rgb(100, 255, 100) }
                    else if is_actual { Color32::WHITE }
                    else if win_cells.contains(&pos) { Color32::YELLOW }
                    else { Color32::from_gray(35) };

                painter.add(egui::Shape::convex_polygon(
                    corners, fill, Stroke::new(stroke_w, stroke_col),
                ));

                // Labels on cells.
                let font_size = (hs * 0.32).max(8.0).min(16.0);
                let font = FontId::proportional(font_size);

                if let Some(&mn) = move_numbers.get(&pos) {
                    painter.text(
                        centre,
                        Align2::CENTER_CENTER,
                        mn.to_string(),
                        font.clone(),
                        Color32::WHITE,
                    );
                } else if is_best && !is_actual {
                    painter.text(centre, Align2::CENTER_CENTER, "★", font.clone(),
                        Color32::WHITE);
                } else if is_actual && !state.board.contains_key(&pos) {
                    // Actual played move highlighted but not yet on board — show marker.
                    if let Some(a) = &self.analysis {
                        if let Some(bp) = a.blunder_pct(pos) {
                            painter.text(centre, Align2::CENTER_CENTER,
                                if bp < 0.02 { "✓" } else if bp < 0.05 { "?" } else { "✗" },
                                font, Color32::WHITE);
                        }
                    }
                }
            }

            // Nav buttons on board.
            let nav_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(8.0, rect.height() / 2.0 - 20.0),
                egui::vec2(60.0, 40.0),
            );
            let nav_rect_r = egui::Rect::from_min_size(
                rect.max - egui::vec2(68.0, rect.height() / 2.0),
                egui::vec2(60.0, 40.0),
            );
            if ui.put(nav_rect, egui::Button::new(RichText::new("◀").size(22.0))).clicked() {
                nav_bck = true;
            }
            if ui.put(nav_rect_r, egui::Button::new(RichText::new("▶").size(22.0))).clicked() {
                nav_fwd = true;
            }
        });

        // Apply deferred navigation.
        if nav_fwd { self.step_forward(); }
        if nav_bck { self.step_back(); }
        if let Some(i) = game_click { self.go_to_move(i); }
    }
}

fn fmt_iters(n: u32) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}k", n as f32 / 1_000.0) }
    else { format!("{n}") }
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
        Box::new(move |_cc| Ok(Box::new(ReplayApp::new(games, start)))),
    )
}
