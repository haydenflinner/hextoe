//! hextoe-train-gui — same training loop as `hextoe-train` with an eframe metrics window.
//!
//! Usage:  cargo run --release --bin hextoe-train-gui [--random-rollout]
//!
//! Hyperparameters: edit `DEFAULT_*` in `hextoe::train` (`src/train.rs`).

use eframe::egui;
use hextoe::train::{run_training, TrainPhase, TrainingConfig, TrainingMonitor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> eframe::Result<()> {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "hextoe-train-gui\n\n\
             Options:\n\
               --random-rollout, -r   MCTS simulations use fast random playouts instead of the NN\n\
               -h, --help             Show this help\n"
        );
        return Ok(());
    }

    let config = TrainingConfig::default_with_cli_rollout();
    let monitor = Arc::new(Mutex::new(TrainingMonitor::new(&config)));
    let cancel = Arc::new(AtomicBool::new(false));

    let mon_thread = monitor.clone();
    let cancel_thread = cancel.clone();
    let cfg_thread = config.clone();
    let train_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let err_slot = train_err.clone();
    thread::spawn(move || {
        let r = run_training(cfg_thread, Some(mon_thread), false, Some(cancel_thread));
        if let Err(e) = r {
            if let Ok(mut g) = err_slot.lock() {
                *g = Some(e.to_string());
            }
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 720.0])
            .with_title("hextoe training"),
        ..Default::default()
    };

    eframe::run_native(
        "hextoe training",
        options,
        Box::new(move |cc| {
            Ok(Box::new(TrainDashboard::new(
                cc,
                monitor,
                cancel,
                train_err,
            )))
        }),
    )
}

struct TrainDashboard {
    monitor: Arc<Mutex<TrainingMonitor>>,
    cancel: Arc<AtomicBool>,
    train_err: Arc<Mutex<Option<String>>>,
}

impl TrainDashboard {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        monitor: Arc<Mutex<TrainingMonitor>>,
        cancel: Arc<AtomicBool>,
        train_err: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self {
            monitor,
            cancel,
            train_err,
        }
    }
}

impl eframe::App for TrainDashboard {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hextoe training");
            ui.label(egui::RichText::new("Same loop as hextoe-train; logs and metrics update live.").weak());

            if let Ok(mut err) = self.train_err.lock() {
                if let Some(msg) = err.take() {
                    ui.colored_label(egui::Color32::RED, format!("Training error: {msg}"));
                }
            }

            ui.separator();

            let snap = self.monitor.lock().ok().map(|g| {
                (
                    g.phase,
                    g.iteration,
                    g.buffer_len,
                    g.buffer_capacity,
                    g.min_buffer_for_training,
                    g.games_per_iter,
                    g.mcts_iters_per_move,
                    g.train_steps,
                    g.batch_size,
                    g.use_random_rollout,
                    g.current_game,
                    g.current_move,
                    g.last_mcts_secs,
                    g.last_self_play_total_secs,
                    g.last_iteration_new_records,
                    g.mean_loss,
                    g.last_checkpoint_msg.clone(),
                )
            });

            if let Some((
                phase,
                iteration,
                buffer_len,
                buffer_cap,
                min_buf,
                games_per_iter,
                mcts_im,
                train_steps,
                batch_size,
                use_random_rollout,
                cur_game,
                cur_move,
                last_mcts,
                sp_total,
                new_recs,
                mean_loss,
                last_ck,
            )) = snap
            {
                ui.columns(2, |cols| {
                    cols[0].group(|ui| {
                        ui.set_min_width(280.0);
                        ui.heading("Run");
                        ui.label(format!("Iteration: {iteration}"));
                        ui.label(format!(
                            "Phase: {}",
                            match phase {
                                TrainPhase::Idle => "idle (between checkpoints)",
                                TrainPhase::SelfPlay => "self-play",
                                TrainPhase::Training => "gradient steps",
                                TrainPhase::SavingCheckpoint => "saving weights",
                            }
                        ));
                        ui.separator();
                        ui.heading("Replay buffer");
                        ui.label(format!("Size: {buffer_len} / {buffer_cap}"));
                        ui.label(format!("Train when ≥ {min_buf} positions"));
                        let fill = if buffer_cap > 0 {
                            buffer_len as f32 / buffer_cap as f32
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(fill.clamp(0.0, 1.0)).show_percentage(),
                        );
                        ui.separator();
                        ui.heading("Hyperparameters");
                        ui.label(format!("Games / iter: {games_per_iter}"));
                        ui.label(format!("MCTS iters / move: {mcts_im}"));
                        ui.label(format!("Train steps / iter: {train_steps}"));
                        ui.label(format!("Batch size: {batch_size}"));
                        ui.label(format!(
                            "MCTS simulation: {}",
                            if use_random_rollout {
                                "random rollout (fast)"
                            } else {
                                "NN policy rollout"
                            }
                        ));
                        ui.separator();
                        ui.heading("Current self-play");
                        ui.label(format!("Game {cur_game} (of {games_per_iter} this iter)"));
                        ui.label(format!("Move {cur_move}"));
                        ui.label(format!("Last MCTS block: {last_mcts:.2}s"));
                        ui.separator();
                        ui.heading("Last iteration");
                        ui.label(format!("Self-play wall time: {sp_total:.1}s"));
                        ui.label(format!("New records: {new_recs}"));
                        if let Some(l) = mean_loss {
                            ui.label(format!("Mean loss: {l:.4}"));
                        }
                        if let Some(ref m) = last_ck {
                            ui.label(egui::RichText::new(m).strong());
                        }
                    });

                    cols[1].group(|ui| {
                        ui.heading("Log");
                        egui::ScrollArea::vertical()
                            .max_height(520.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                if let Ok(g) = self.monitor.lock() {
                                    for line in &g.log {
                                        ui.monospace(line);
                                    }
                                }
                            });
                    });
                });
            }

            ui.separator();
            if ui.button("Stop training").clicked() {
                self.cancel.store(true, Ordering::Relaxed);
            }
            ui.small("Request cooperative shutdown (finishes current work units where possible).");
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(120));
    }
}
