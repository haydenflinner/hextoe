use hextoe::api::{start_api_server, GameUpdate};
use hextoe::device::default_inference_device;
use hextoe::game::{winning_line, GameState, Player, Pos};
use hextoe::mcts::{Mcts, RandomRollout};
use hextoe::nn::{LoadedNet, NeuralRollout};
use hextoe::nnue::{NNUERollout, NNUEWeights, DEFAULT_NNUE_PATH};
use hextoe::train::default_inference_checkpoint_path;

use candle_core::Device as CandleDevice;
use eframe::egui;
use egui::text::{Galley, LayoutJob};
use egui::{Align, Color32, FontId, Pos2, RichText, Stroke};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

const HEX_SIZE: f32 = 32.0;
/// Iterations per batch sent by the background thread.
const BATCH_SIZE: u32 = 300;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RolloutMode {
    Random,
    Neural,
    Nnue,
}

impl RolloutMode {
    fn label(self) -> &'static str {
        match self {
            RolloutMode::Random => "Random rollout",
            RolloutMode::Neural => "NN-based rollout",
            RolloutMode::Nnue => "NNUE rollout",
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1060.0, 740.0])
            .with_title("Hextoe"),
        ..Default::default()
    };
    eframe::run_native(
        "Hextoe",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

// ── Application state ─────────────────────────────────────────────────────────

struct App {
    game: GameState,
    /// A training checkpoint exists (latest / best / legacy); each MCTS thread loads its own copy.
    nn_checkpoint_hint: bool,
    /// An NNUE checkpoint exists at DEFAULT_NNUE_PATH.
    nnue_checkpoint_hint: bool,
    /// Current best suggestions: (pos, score 0–1, visits, policy_share).
    suggestions: Vec<(Pos, f32, u32, f32)>,
    /// Total MCTS iterations accumulated since the last move.
    mcts_iters: u32,
    /// Signal the background thread to stop.
    cancel: Arc<AtomicBool>,
    /// Background thread sends (suggestions, total_iters) as it accumulates.
    result_rx: Option<Receiver<(Vec<(Pos, f32, u32, f32)>, u32)>>,
    last_pos: Option<Pos>,
    pan_offset: egui::Vec2,
    rollout_mode: RolloutMode,
    /// Receives game state updates from the REST API (bookmarklet).
    api_rx: Receiver<GameUpdate>,
    /// When the last API message arrived (for the activity flash).
    last_api_msg: Option<Instant>,
}

impl App {
    fn new() -> Self {
        let nn_checkpoint_hint = default_inference_checkpoint_path().is_some();
        let nnue_checkpoint_hint = std::path::Path::new(DEFAULT_NNUE_PATH).exists();
        let api_rx = start_api_server();
        let mut app = App {
            game: GameState::new(),
            nn_checkpoint_hint,
            nnue_checkpoint_hint,
            suggestions: vec![],
            mcts_iters: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            result_rx: None,
            last_pos: None,
            pan_offset: egui::Vec2::ZERO,
            rollout_mode: if nnue_checkpoint_hint {
                RolloutMode::Nnue
            } else if nn_checkpoint_hint {
                RolloutMode::Neural
            } else {
                RolloutMode::Random
            },
            api_rx,
            last_api_msg: None,
        };
        app.restart_mcts();
        app
    }

    /// Cancel any running background thread and start a new one for the
    /// current game state.  The thread runs continuously until cancelled,
    /// improving suggestions with each batch.
    fn restart_mcts(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.mcts_iters = 0;
        self.result_rx = None;

        if self.game.is_terminal() {
            return;
        }

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = Arc::clone(&cancel);

        let (tx, rx) = mpsc::channel();
        let game = self.game.clone();
        let rollout_mode = self.rollout_mode;

        thread::spawn(move || {
            let device = default_inference_device();
            let loaded_nn = if rollout_mode == RolloutMode::Neural {
                default_inference_checkpoint_path()
                    .and_then(|p| LoadedNet::try_load(p, &device).ok())
            } else {
                None
            };
            let nnue_weights = if rollout_mode == RolloutMode::Nnue {
                NNUEWeights::load(DEFAULT_NNUE_PATH, &CandleDevice::Cpu).ok().map(Arc::new)
            } else {
                None
            };
            let mut rng = rand::thread_rng();
            let mut mcts = Mcts::new(game);
            loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                match (rollout_mode, &loaded_nn, &nnue_weights) {
                    (RolloutMode::Neural, Some(ld), _) => {
                        let r = NeuralRollout {
                            net: &ld.net,
                            device: &device,
                        };
                        mcts.search_iters(BATCH_SIZE, &mut rng, &r);
                    }
                    (RolloutMode::Nnue, _, Some(w)) => {
                        let r = NNUERollout::new(Arc::clone(w));
                        mcts.search_iters(BATCH_SIZE, &mut rng, &r);
                    }
                    _ => {
                        let r = RandomRollout;
                        mcts.search_iters(BATCH_SIZE, &mut rng, &r);
                    }
                }
                let iters = mcts.total_visits();
                let best = mcts.best_moves(3);
                if tx.send((best, iters)).is_err() {
                    break;
                }
            }
        });

        self.result_rx = Some(rx);
    }

    fn handle_click(&mut self, pos: Pos) {
        if self.game.is_terminal() {
            return;
        }
        if self.game.place(pos) {
            self.last_pos = Some(pos);
            self.suggestions.clear();
            self.restart_mcts();
        }
    }

    fn reset(&mut self) {
        self.game = GameState::new();
        self.suggestions.clear();
        self.last_pos = None;
        self.pan_offset = egui::Vec2::ZERO;
        self.restart_mcts();
    }
}

// ── Hex geometry ──────────────────────────────────────────────────────────────

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
    let rq = fq.round();
    let rr = fr.round();
    let rs = fs.round();
    let dq = (rq - fq).abs();
    let dr = (rr - fr).abs();
    let ds = (rs - fs).abs();
    if dq > dr && dq > ds {
        ((-rr - rs) as i32, rr as i32)
    } else if dr > ds {
        (rq as i32, (-rq - rs) as i32)
    } else {
        (rq as i32, rr as i32)
    }
}

fn hex_corners(centre: Pos2, size: f32) -> [Pos2; 6] {
    std::array::from_fn(|i| {
        let angle = std::f32::consts::FRAC_PI_6 + std::f32::consts::FRAC_PI_3 * i as f32;
        Pos2::new(centre.x + size * angle.cos(), centre.y + size * angle.sin())
    })
}

/// Horizontal half-width available for text at vertical offset `dy` from hex centre
/// (positive `dy` = downward). `r` is the same circumradius as `hex_corners(_, r)`.
fn hex_max_text_half_width(r: f32, dy: f32) -> f32 {
    let abs_y = dy.abs();
    if abs_y <= r * 0.5 {
        r * 3f32.sqrt() * 0.5
    } else if abs_y <= r {
        (r - abs_y) * 3f32.sqrt()
    } else {
        0.0
    }
}

fn layout_centered_suggestion_galley(
    painter: &egui::Painter,
    text: String,
    font_id: FontId,
) -> Arc<Galley> {
    let mut job = LayoutJob::simple(text, font_id, Color32::BLACK, f32::INFINITY);
    job.halign = Align::Center;
    painter.layout_job(job)
}

/// Shrink font until each row fits the hex width at that height and the block fits vertically.
fn fit_suggestion_galley_in_hex(
    painter: &egui::Painter,
    text: String,
    r: f32,
    initial_font_px: f32,
) -> (f32, Arc<Galley>) {
    const MIN_FONT: f32 = 4.0;
    const MARGIN: f32 = 0.90;
    const MAX_ITERS: usize = 10;

    let mut font_size = initial_font_px;
    for _ in 0..MAX_ITERS {
        let font_id = FontId::proportional(font_size);
        let galley = layout_centered_suggestion_galley(painter, text.clone(), font_id);
        let mut scale = 1.0_f32;

        let cy = galley.rect.center().y;
        for row in &galley.rows {
            let dy = row.rect.center().y - cy;
            let max_w = 2.0 * hex_max_text_half_width(r, dy) * MARGIN;
            if row.rect.width() > 0.0 && max_w > 0.0 && row.rect.width() > max_w {
                scale = scale.min(max_w / row.rect.width());
            }
        }

        let max_h = 2.0 * r * MARGIN;
        if galley.rect.height() > max_h {
            scale = scale.min(max_h / galley.rect.height());
        }

        if scale >= 0.999 {
            return (font_size, galley);
        }
        font_size = (font_size * scale).max(MIN_FONT);
    }
    let font_id = FontId::proportional(font_size);
    let galley = layout_centered_suggestion_galley(painter, text, font_id);
    (font_size, galley)
}

// ── Colour helpers ────────────────────────────────────────────────────────────

fn player_color(p: Player) -> Color32 {
    match p {
        Player::X => Color32::from_rgb(70, 140, 220),
        Player::O => Color32::from_rgb(220, 70, 70),
    }
}

fn player_dark(p: Player) -> Color32 {
    match p {
        Player::X => Color32::from_rgb(45, 95, 165),
        Player::O => Color32::from_rgb(165, 45, 45),
    }
}

fn cell_fill(
    pos: Pos,
    game: &GameState,
    win_cells: &std::collections::HashSet<Pos>,
    suggestion_rank: Option<usize>,
    hovered: Option<Pos>,
) -> Color32 {
    if let Some(player) = game.board.get(&pos) {
        return if win_cells.contains(&pos) {
            match player {
                Player::X => Color32::from_rgb(110, 195, 255),
                Player::O => Color32::from_rgb(255, 120, 120),
            }
        } else {
            player_dark(*player)
        };
    }
    if Some(pos) == hovered && !game.is_terminal() {
        return Color32::from_gray(95);
    }
    if let Some(rank) = suggestion_rank {
        return match rank {
            0 => Color32::from_rgb(45, 185, 45),
            1 => Color32::from_rgb(130, 190, 45),
            _ => Color32::from_rgb(175, 190, 45),
        };
    }
    Color32::from_gray(52)
}

// ── egui App ──────────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain all pending MCTS results — keep the latest.
        if let Some(rx) = &self.result_rx {
            let mut latest = None;
            while let Ok(msg) = rx.try_recv() {
                latest = Some(msg);
            }
            if let Some((best, iters)) = latest {
                self.suggestions = best;
                self.mcts_iters = iters;
            }
            ctx.request_repaint(); // keep polling every frame
        }

        // Drain REST API updates from the bookmarklet.
        while let Ok(update) = self.api_rx.try_recv() {
            self.last_api_msg = Some(Instant::now());
            match update {
                GameUpdate::Reset(state) => {
                    self.game = state;
                    self.last_pos = None;
                    self.suggestions.clear();
                    self.restart_mcts();
                }
                GameUpdate::Move(pos) => {
                    if self.game.place(pos) {
                        self.last_pos = Some(pos);
                        self.suggestions.clear();
                        self.restart_mcts();
                    }
                }
            }
            ctx.request_repaint();
        }

        // ── Side panel ────────────────────────────────────────────────────
        egui::SidePanel::right("side")
            .exact_width(230.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);

                // ── Prominent player indicator ─────────────────────────────
                if let Some(winner) = self.game.winner {
                    egui::Frame::none()
                        .fill(player_color(winner))
                        .inner_margin(egui::Margin::symmetric(10.0, 14.0))
                        .rounding(egui::Rounding::same(8.0))
                        .show(ui, |ui| {
                            ui.with_layout(
                                egui::Layout::top_down(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(format!("{winner:?} wins!"))
                                            .size(26.0)
                                            .color(Color32::WHITE)
                                            .strong(),
                                    );
                                },
                            );
                        });
                } else {
                    let player = self.game.current_player();
                    let col = player_color(player);
                    egui::Frame::none()
                        .fill(col)
                        .inner_margin(egui::Margin::symmetric(10.0, 10.0))
                        .rounding(egui::Rounding::same(8.0))
                        .show(ui, |ui| {
                            ui.with_layout(
                                egui::Layout::top_down(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(format!("{player:?} to move"))
                                            .size(22.0)
                                            .color(Color32::WHITE)
                                            .strong(),
                                    );
                                    ui.label(
                                        RichText::new(self.game.turn_label())
                                            .size(14.0)
                                            .color(Color32::from_rgba_unmultiplied(
                                                255, 255, 255, 200,
                                            )),
                                    );
                                },
                            );
                        });
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.label(RichText::new("Analysis mode").strong());
                let mut next_mode = self.rollout_mode;
                ui.radio_value(&mut next_mode, RolloutMode::Random, RolloutMode::Random.label());
                ui.radio_value(&mut next_mode, RolloutMode::Neural, RolloutMode::Neural.label());
                ui.radio_value(&mut next_mode, RolloutMode::Nnue, RolloutMode::Nnue.label());
                if next_mode != self.rollout_mode {
                    self.rollout_mode = next_mode;
                    self.suggestions.clear();
                    self.restart_mcts();
                }
                if self.rollout_mode == RolloutMode::Neural && !self.nn_checkpoint_hint {
                    ui.label(
                        RichText::new("No checkpoint found: falling back to random rollout.")
                            .size(11.0)
                            .color(Color32::from_rgb(200, 160, 90)),
                    );
                }
                if self.rollout_mode == RolloutMode::Nnue && !self.nnue_checkpoint_hint {
                    ui.label(
                        RichText::new("No NNUE checkpoint found: falling back to random rollout.")
                            .size(11.0)
                            .color(Color32::from_rgb(200, 160, 90)),
                    );
                }
                ui.add_space(4.0);

                // ── MCTS suggestions ───────────────────────────────────────
                if self.game.is_terminal() {
                    // nothing
                } else if self.suggestions.is_empty() {
                    ui.label("Analysing…");
                    ui.spinner();
                } else {
                    if self.rollout_mode == RolloutMode::Neural && self.nn_checkpoint_hint {
                        ui.label(
                            RichText::new(
                                "Analysis: NN value at leaves (AlphaZero-style; not MC win %)",
                            )
                            .size(12.0)
                            .color(Color32::from_rgb(120, 200, 140)),
                        );
                        ui.add_space(4.0);
                    } else if self.rollout_mode == RolloutMode::Nnue && self.nnue_checkpoint_hint {
                        ui.label(
                            RichText::new(
                                "Analysis: NNUE value + tactical priors (fast CPU eval)",
                            )
                            .size(12.0)
                            .color(Color32::from_rgb(120, 200, 200)),
                        );
                        ui.add_space(4.0);
                    } else {
                        ui.label(
                            RichText::new(
                                "No checkpoint — random rollouts to terminal (MC win estimate)",
                            )
                            .size(11.0)
                            .color(Color32::from_rgb(200, 160, 90)),
                        );
                        ui.add_space(4.0);
                    }
                    ui.label(format!(
                        "Top moves — score 0–100% ({} iters)",
                        fmt_iters(self.mcts_iters)
                    ));
                    ui.add_space(2.0);
                    for (rank, (pos, wr, visits, pol)) in self.suggestions.iter().enumerate() {
                        let marker = ["①", "②", "③"][rank];
                        let col = match rank {
                            0 => Color32::from_rgb(45, 185, 45),
                            1 => Color32::from_rgb(130, 190, 45),
                            _ => Color32::from_rgb(175, 190, 45),
                        };
                        ui.colored_label(
                            col,
                            format!(
                                "{marker} ({},{})  {:.0}%  ·  {} visits  ·  {:.1}% policy",
                                pos.0,
                                pos.1,
                                wr * 100.0,
                                fmt_iters(*visits),
                                pol * 100.0
                            ),
                        );
                    }
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.label(
                    RichText::new("Rules").strong(),
                );
                ui.label("• 6 in a row wins");
                ui.label("• X plays 1 anchor move");
                ui.label("• Then pairs: OO XX OO…");

                ui.add_space(4.0);
                ui.label(RichText::new("Controls").strong());
                ui.label("• Click to place a piece");
                ui.label("• Drag to pan the board");

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(4.0);

                // ── REST API status ────────────────────────────────────────
                {
                    const FADE_SECS: f32 = 2.0;
                    let age = self.last_api_msg
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(f32::MAX);
                    // Dot colour: bright green → dim green over FADE_SECS.
                    let (dot_col, label) = if age < FADE_SECS {
                        let t = 1.0 - (age / FADE_SECS);
                        let g = (55.0 + 200.0 * t) as u8;
                        (Color32::from_rgb(30, g, 60), "API: received")
                    } else {
                        (Color32::from_rgb(30, 80, 50), "API: listening :8080")
                    };
                    ui.horizontal(|ui| {
                        // Filled circle as activity dot.
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(10.0, 10.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().circle_filled(rect.center(), 5.0, dot_col);
                        ui.label(RichText::new(label).size(11.0).color(dot_col));
                    });
                    // Keep repainting while the flash is still fading.
                    if age < FADE_SECS {
                        ctx.request_repaint();
                    }
                }

                ui.add_space(4.0);
                ui.separator();
                if ui
                    .add_sized([210.0, 32.0], egui::Button::new("New Game"))
                    .clicked()
                {
                    self.reset();
                }
            });

        // ── Board panel ───────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let origin = rect.center() + self.pan_offset;

            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            if response.dragged() {
                self.pan_offset += response.drag_delta();
            }

            // Determine winning line.
            let win_cells: std::collections::HashSet<Pos> =
                if let (Some(w), Some(last)) = (self.game.winner, self.last_pos) {
                    winning_line(&self.game.board, last, w).into_iter().collect()
                } else {
                    Default::default()
                };

            let suggestion_map: std::collections::HashMap<Pos, (usize, f32, u32, f32)> = self
                .suggestions
                .iter()
                .enumerate()
                .map(|(i, &(p, wr, v, pol))| (p, (i, wr, v, pol)))
                .collect();

            // Cells to render.
            let mut cells: std::collections::HashSet<Pos> =
                self.game.board.keys().copied().collect();
            if cells.is_empty() {
                for dq in -3i32..=3 {
                    for dr in (-3i32).max(-dq - 3)..=3i32.min(-dq + 3) {
                        cells.insert((dq, dr));
                    }
                }
            }
            for &p in &self.game.candidates {
                cells.insert(p);
            }
            for &(p, _, _, _) in &self.suggestions {
                cells.insert(p);
            }

            let painter = ui.painter_at(rect);

            let hovered = ctx.pointer_hover_pos().and_then(|p| {
                if rect.contains(p) && !self.game.is_terminal() {
                    Some(pixel_to_hex(p, HEX_SIZE, origin))
                } else {
                    None
                }
            });

            let expanded = rect.expand(HEX_SIZE * 3.0);
            for pos in &cells {
                let centre = hex_to_pixel(pos.0, pos.1, HEX_SIZE, origin);
                if !expanded.contains(centre) {
                    continue;
                }

                let sug_rank = suggestion_map.get(pos).map(|&(r, _, _, _)| r);
                let corners: Vec<Pos2> = hex_corners(centre, HEX_SIZE - 1.5).into();
                let fill = cell_fill(*pos, &self.game, &win_cells, sug_rank, hovered);
                let stroke_col = if win_cells.contains(pos) {
                    Color32::YELLOW
                } else {
                    Color32::from_gray(35)
                };

                painter.add(egui::Shape::convex_polygon(
                    corners,
                    fill,
                    Stroke::new(if win_cells.contains(pos) { 2.5 } else { 1.0 }, stroke_col),
                ));

                // Labels.
                if let Some(player) = self.game.board.get(pos) {
                    painter.text(
                        centre,
                        egui::Align2::CENTER_CENTER,
                        if *player == Player::X { "X" } else { "O" },
                        FontId::proportional(HEX_SIZE * 0.65),
                        Color32::WHITE,
                    );
                } else if let Some(&(rank, wr, visits, pol)) = suggestion_map.get(pos) {
                    let marker = ["①", "②", "③"][rank];
                    let label = format!(
                        "{marker}\n{:.0}%\n{} · {:.0}%",
                        wr * 100.0,
                        fmt_iters(visits),
                        pol * 100.0
                    );
                    let r = HEX_SIZE - 1.5;
                    let (_font_px, galley) =
                        fit_suggestion_galley_in_hex(&painter, label, r, HEX_SIZE * 0.32);
                    let draw_pos = centre - galley.rect.center().to_vec2();
                    painter.galley(draw_pos, galley, Color32::BLACK);
                }
            }

            // Handle click.
            if response.clicked() {
                if let Some(click) = response.interact_pointer_pos() {
                    self.handle_click(pixel_to_hex(click, HEX_SIZE, origin));
                }
            }
        });
    }
}

fn fmt_iters(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f32 / 1_000.0)
    } else {
        format!("{n}")
    }
}
