mod game;
mod mcts;

use eframe::egui;
use egui::{Color32, FontId, Pos2, Stroke};
use game::{winning_line, GameState, Player, Pos};
use mcts::Mcts;
use std::sync::mpsc::{self, Receiver};
use std::thread;

const HEX_SIZE: f32 = 32.0;
const MCTS_ITERATIONS: u32 = 4_000;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 720.0])
            .with_title("Hextoe"),
        ..Default::default()
    };
    eframe::run_native(
        "Hextoe",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

// ── Application state ────────────────────────────────────────────────────────

struct App {
    game: GameState,
    /// Top-N MCTS suggestions: (position, win_rate in [0,1])
    suggestions: Vec<(Pos, f32)>,
    computing: bool,
    result_rx: Option<Receiver<Vec<(Pos, f32)>>>,
    /// Position of the last placed piece (for win-line detection).
    last_pos: Option<Pos>,
    /// Offset applied to the board view (panning).
    pan_offset: egui::Vec2,
    drag_start: Option<Pos2>,
}

impl App {
    fn new() -> Self {
        let mut app = App {
            game: GameState::new(),
            suggestions: vec![],
            computing: false,
            result_rx: None,
            last_pos: None,
            pan_offset: egui::Vec2::ZERO,
            drag_start: None,
        };
        app.start_mcts();
        app
    }

    fn start_mcts(&mut self) {
        if self.game.is_terminal() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let game_clone = self.game.clone();
        thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let mut mcts = Mcts::new(game_clone);
            let _ = tx.send(mcts.search(MCTS_ITERATIONS, &mut rng));
        });
        self.result_rx = Some(rx);
        self.computing = true;
    }

    fn handle_click(&mut self, pos: Pos) {
        if self.game.is_terminal() {
            return;
        }
        if self.game.place(pos) {
            self.last_pos = Some(pos);
            self.suggestions.clear();
            // Drop any running computation and start a new one.
            self.result_rx = None;
            self.computing = false;
            if !self.game.is_terminal() {
                self.start_mcts();
            }
        }
    }

    fn reset(&mut self) {
        self.game = GameState::new();
        self.suggestions.clear();
        self.computing = false;
        self.result_rx = None;
        self.last_pos = None;
        self.pan_offset = egui::Vec2::ZERO;
        self.start_mcts();
    }
}

// ── Hex geometry helpers ─────────────────────────────────────────────────────

fn hex_to_pixel(q: i32, r: i32, size: f32, origin: Pos2) -> Pos2 {
    Pos2::new(
        origin.x + size * (3f32.sqrt() * q as f32 + 3f32.sqrt() / 2.0 * r as f32),
        origin.y + size * (1.5 * r as f32),
    )
}

fn pixel_to_hex(p: Pos2, size: f32, origin: Pos2) -> Pos {
    let x = p.x - origin.x;
    let y = p.y - origin.y;
    let fq = (3f32.sqrt() / 3.0 * x - 1.0 / 3.0 * y) / size;
    let fr = 2.0 / 3.0 * y / size;
    hex_round(fq, fr)
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

/// Six corners of a pointy-top hexagon centred at `centre`.
fn hex_corners(centre: Pos2, size: f32) -> [Pos2; 6] {
    std::array::from_fn(|i| {
        let angle = std::f32::consts::FRAC_PI_6 + std::f32::consts::FRAC_PI_3 * i as f32;
        Pos2::new(centre.x + size * angle.cos(), centre.y + size * angle.sin())
    })
}

// ── egui App impl ─────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll MCTS background thread.
        if let Some(rx) = &self.result_rx {
            if let Ok(results) = rx.try_recv() {
                self.suggestions = results.into_iter().take(3).collect();
                self.computing = false;
                self.result_rx = None;
            } else {
                ctx.request_repaint(); // keep polling
            }
        }

        // ── Side panel ───────────────────────────────────────────────────────
        egui::SidePanel::right("side")
            .min_width(210.0)
            .max_width(210.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Hextoe");
                ui.separator();

                if let Some(winner) = self.game.winner {
                    let color = player_color(winner);
                    ui.colored_label(color, format!("{:?} wins!", winner));
                } else {
                    let player = self.game.current_player();
                    ui.colored_label(
                        player_color(player),
                        format!(
                            "{:?}'s turn — {}",
                            player,
                            self.game.turn_label()
                        ),
                    );
                }

                ui.separator();

                if self.computing {
                    ui.label("Computing suggestions…");
                    ui.spinner();
                } else if self.suggestions.is_empty() && !self.game.is_terminal() {
                    ui.label("No suggestions yet.");
                } else {
                    ui.label("Top suggestions:");
                    for (rank, (pos, wr)) in self.suggestions.iter().enumerate() {
                        let marker = ["①", "②", "③"][rank];
                        ui.label(format!(
                            "{} ({},{})  {:.0}%",
                            marker,
                            pos.0,
                            pos.1,
                            wr * 100.0
                        ));
                    }
                }

                ui.separator();
                ui.label("Rules");
                ui.label("• 6 in a row wins");
                ui.label("• X plays 1 anchor move");
                ui.label("• Then: OO XX OO XX …");
                ui.separator();
                ui.label("Controls");
                ui.label("• Click to place");
                ui.label("• Drag to pan");

                ui.separator();
                if ui.button("New Game").clicked() {
                    self.reset();
                }
            });

        // ── Central panel (board) ─────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let origin = rect.center() + self.pan_offset;

            // Handle pan (drag).
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            if response.drag_started() {
                self.drag_start = ctx.pointer_interact_pos();
            }
            if response.dragged() {
                self.pan_offset += response.drag_delta();
            }

            // Determine cells to render.
            let mut cells: std::collections::HashSet<Pos> = self.game.board.keys().copied().collect();

            if cells.is_empty() {
                // Initial view: small grid around origin.
                for dq in -3i32..=3 {
                    let dr_lo = (-3i32).max(-dq - 3);
                    let dr_hi = 3i32.min(-dq + 3);
                    for dr in dr_lo..=dr_hi {
                        cells.insert((dq, dr));
                    }
                }
            }

            // Show candidate cells (valid next moves) as empty hexes.
            for &p in &self.game.candidates {
                cells.insert(p);
            }

            // Winning line cells.
            let win_cells: std::collections::HashSet<Pos> =
                if let (Some(winner), Some(last)) = (self.game.winner, self.last_pos) {
                    winning_line(&self.game.board, last, winner)
                        .into_iter()
                        .collect()
                } else {
                    std::collections::HashSet::new()
                };

            let suggestion_map: std::collections::HashMap<Pos, (usize, f32)> = self
                .suggestions
                .iter()
                .enumerate()
                .map(|(i, &(p, wr))| (p, (i, wr)))
                .collect();

            let painter = ui.painter_at(rect);

            // Draw hover highlight.
            let hovered_hex = ctx.pointer_hover_pos().and_then(|p| {
                if rect.contains(p) && !self.game.is_terminal() {
                    Some(pixel_to_hex(p, HEX_SIZE, origin))
                } else {
                    None
                }
            });

            for pos in &cells {
                let centre = hex_to_pixel(pos.0, pos.1, HEX_SIZE, origin);
                // Cull cells outside the visible rect (with margin).
                if !rect.expand(HEX_SIZE * 2.0).contains(centre) {
                    continue;
                }
                let corners: Vec<Pos2> = hex_corners(centre, HEX_SIZE - 1.0).into();

                let fill = cell_fill(
                    *pos,
                    &self.game,
                    &win_cells,
                    &suggestion_map,
                    hovered_hex,
                );
                let stroke_color = if win_cells.contains(pos) {
                    Color32::YELLOW
                } else {
                    Color32::from_gray(40)
                };
                let stroke_width = if win_cells.contains(pos) { 2.5 } else { 1.0 };

                painter.add(egui::Shape::convex_polygon(
                    corners,
                    fill,
                    Stroke::new(stroke_width, stroke_color),
                ));

                // Label: piece or suggestion percentage.
                match self.game.board.get(pos) {
                    Some(p) => {
                        let label = if *p == Player::X { "X" } else { "O" };
                        painter.text(
                            centre,
                            egui::Align2::CENTER_CENTER,
                            label,
                            FontId::proportional(HEX_SIZE * 0.65),
                            Color32::WHITE,
                        );
                    }
                    None => {
                        if let Some(&(rank, wr)) = suggestion_map.get(pos) {
                            let marker = ["①", "②", "③"][rank];
                            painter.text(
                                centre,
                                egui::Align2::CENTER_CENTER,
                                format!("{}\n{:.0}%", marker, wr * 100.0),
                                FontId::proportional(HEX_SIZE * 0.38),
                                Color32::BLACK,
                            );
                        }
                    }
                }
            }

            // Handle click (not drag).
            if response.clicked() {
                if let Some(click_pos) = response.interact_pointer_pos() {
                    let hex = pixel_to_hex(click_pos, HEX_SIZE, origin);
                    self.handle_click(hex);
                }
            }
        });
    }
}

// ── Colour helpers ────────────────────────────────────────────────────────────

fn player_color(p: Player) -> Color32 {
    match p {
        Player::X => Color32::from_rgb(100, 160, 230),
        Player::O => Color32::from_rgb(230, 90, 90),
    }
}

fn cell_fill(
    pos: Pos,
    game: &GameState,
    win_cells: &std::collections::HashSet<Pos>,
    suggestion_map: &std::collections::HashMap<Pos, (usize, f32)>,
    hovered: Option<Pos>,
) -> Color32 {
    if let Some(player) = game.board.get(&pos) {
        if win_cells.contains(&pos) {
            // Winning piece: brighter highlight.
            return match player {
                Player::X => Color32::from_rgb(60, 180, 255),
                Player::O => Color32::from_rgb(255, 80, 80),
            };
        }
        return match player {
            Player::X => Color32::from_rgb(60, 110, 175),
            Player::O => Color32::from_rgb(185, 55, 55),
        };
    }

    if Some(pos) == hovered && !game.is_terminal() {
        return Color32::from_gray(90);
    }

    if let Some(&(rank, _)) = suggestion_map.get(&pos) {
        return match rank {
            0 => Color32::from_rgb(60, 185, 60),
            1 => Color32::from_rgb(140, 195, 60),
            _ => Color32::from_rgb(185, 195, 60),
        };
    }

    Color32::from_gray(55)
}
