//! Shared self-play training loop (AlphaZero-style) used by `hextoe-train` and `hextoe-train-gui`.

use candle_core::{Device, DType, Tensor};
use candle_nn::{optim::AdamW, optim::ParamsAdamW, Optimizer, VarBuilder, VarMap};
use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::SeedableRng;

use crate::encode::{CHANNELS, GRID};
use crate::game::Player;
use crate::mcts::{RandomRollout, RolloutPolicy};
use crate::nn::{load_weights, save_weights, HextoeNet, LoadedNet, NeuralRollout};
use crate::self_play::{ReplayBuffer, SelfPlayCollector};

const LOG_CAP: usize = 400;

// ── Default hyperparameters (single source of truth for hextoe-train / hextoe-train-gui) ──

/// Weights after each training step (always overwritten).
pub const DEFAULT_LATEST_PATH: &str = "hextoe_model_latest.safetensors";
/// Champion weights used as the promotion gate opponent.
pub const DEFAULT_BEST_PATH: &str = "hextoe_model_best.safetensors";
/// Legacy single-file checkpoint (loaded if latest/best are missing).
pub const DEFAULT_LEGACY_MODEL_PATH: &str = "hextoe_model.safetensors";
pub const DEFAULT_REPLAY_CAPACITY: usize = 50_000;
pub const DEFAULT_MIN_BUFFER_FOR_TRAINING: usize = 500;
/// Wall-clock budget for self-play before each training step.
pub const DEFAULT_SELF_PLAY_SECS: f64 = 30.0;
/// Wall-clock budget for new-vs-best games after training (NN mode only).
pub const DEFAULT_PROMOTION_EVAL_SECS: f64 = 15.0;
/// Promote `latest` → `best` if new wins at least this fraction of eval games (no draws counted in denominator).
pub const DEFAULT_PROMOTION_MIN_WIN_RATE: f64 = 0.52;
pub const DEFAULT_MCTS_ITERS_PER_MOVE: u32 = 200;
pub const DEFAULT_TRAIN_STEPS: usize = 5;
pub const DEFAULT_BATCH_SIZE: usize = 128;
pub const DEFAULT_LR: f64 = 3e-4;
pub const DEFAULT_WEIGHT_DECAY: f64 = 1e-4;
pub const DEFAULT_SELF_PLAY_PROGRESS_EVERY_N_MOVES: u32 = 1;

/// Hyperparameters for the training loop.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub latest_path: String,
    pub best_path: String,
    pub replay_capacity: usize,
    pub min_buffer_for_training: usize,
    /// Seconds of wall-clock self-play per iteration before training.
    pub self_play_secs: f64,
    /// Seconds of new-vs-best games for the promotion gate (after each training step).
    pub promotion_eval_secs: f64,
    pub promotion_min_win_rate: f64,
    pub mcts_iters_per_move: u32,
    pub train_steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub self_play_progress_every_n_moves: u32,
    pub device: Device,
    /// If true, MCTS simulations use fast uniform random playouts instead of the NN policy.
    pub use_random_rollout: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::from_defaults(false)
    }
}

impl TrainingConfig {
    /// Build from [`DEFAULT_*`] constants; `use_random_rollout` is passed explicitly.
    pub fn from_defaults(use_random_rollout: bool) -> Self {
        Self {
            latest_path: DEFAULT_LATEST_PATH.to_string(),
            best_path: DEFAULT_BEST_PATH.to_string(),
            replay_capacity: DEFAULT_REPLAY_CAPACITY,
            min_buffer_for_training: DEFAULT_MIN_BUFFER_FOR_TRAINING,
            self_play_secs: DEFAULT_SELF_PLAY_SECS,
            promotion_eval_secs: DEFAULT_PROMOTION_EVAL_SECS,
            promotion_min_win_rate: DEFAULT_PROMOTION_MIN_WIN_RATE,
            mcts_iters_per_move: DEFAULT_MCTS_ITERS_PER_MOVE,
            train_steps: DEFAULT_TRAIN_STEPS,
            batch_size: DEFAULT_BATCH_SIZE,
            lr: DEFAULT_LR,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            self_play_progress_every_n_moves: DEFAULT_SELF_PLAY_PROGRESS_EVERY_N_MOVES,
            device: Device::Cpu,
            use_random_rollout,
        }
    }

    /// Same as [`TrainingConfig::default`], but sets `use_random_rollout` from `--random-rollout` / `-r`.
    pub fn default_with_cli_rollout() -> Self {
        Self::from_defaults(cli_use_random_rollout())
    }
}

/// True if argv contains `--random-rollout` or `-r` (for `hextoe-train` / `hextoe-train-gui`).
pub fn cli_use_random_rollout() -> bool {
    std::env::args().skip(1).any(|a| a == "--random-rollout" || a == "-r")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainPhase {
    Idle,
    SelfPlay,
    Training,
    SavingCheckpoint,
    PromotionEval,
}

/// Live snapshot for UIs (`Arc<Mutex<TrainingMonitor>>`).
#[derive(Debug)]
pub struct TrainingMonitor {
    pub phase: TrainPhase,
    pub iteration: u32,
    pub buffer_len: usize,
    pub buffer_capacity: usize,
    pub min_buffer_for_training: usize,
    pub self_play_secs: f64,
    pub promotion_eval_secs: f64,
    pub games_this_iter: usize,
    pub mcts_iters_per_move: u32,
    pub train_steps: usize,
    pub batch_size: usize,
    pub current_game: usize,
    pub current_move: u32,
    pub last_mcts_secs: f64,
    pub last_self_play_total_secs: f64,
    pub last_iteration_new_records: usize,
    pub mean_loss: Option<f32>,
    pub last_checkpoint_msg: Option<String>,
    pub last_promotion_msg: Option<String>,
    /// MCTS uses [`RandomRollout`] when true, [`NeuralRollout`] when false.
    pub use_random_rollout: bool,
    pub log: VecDeque<String>,
}

impl TrainingMonitor {
    pub fn new(config: &TrainingConfig) -> Self {
        TrainingMonitor {
            phase: TrainPhase::Idle,
            iteration: 0,
            buffer_len: 0,
            buffer_capacity: config.replay_capacity,
            min_buffer_for_training: config.min_buffer_for_training,
            self_play_secs: config.self_play_secs,
            promotion_eval_secs: config.promotion_eval_secs,
            games_this_iter: 0,
            mcts_iters_per_move: config.mcts_iters_per_move,
            train_steps: config.train_steps,
            batch_size: config.batch_size,
            current_game: 0,
            current_move: 0,
            last_mcts_secs: 0.0,
            last_self_play_total_secs: 0.0,
            last_iteration_new_records: 0,
            mean_loss: None,
            last_checkpoint_msg: None,
            last_promotion_msg: None,
            use_random_rollout: config.use_random_rollout,
            log: VecDeque::with_capacity(LOG_CAP.min(32)),
        }
    }

    fn push_line(&mut self, line: impl Into<String>) {
        let s = line.into();
        if self.log.len() >= LOG_CAP {
            self.log.pop_front();
        }
        self.log.push_back(s);
    }
}

fn log_line(
    monitor: &Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    line: &str,
) {
    if log_stdout {
        println!("{line}");
        let _ = std::io::stdout().flush();
    }
    if let Some(m) = monitor {
        if let Ok(mut g) = m.lock() {
            g.push_line(line.to_string());
        }
    }
}

/// Play full games until `self_play_secs` wall time has elapsed (no new game starts after the budget).
fn self_play_until_duration<P: RolloutPolicy>(
    collector: &SelfPlayCollector,
    config: &TrainingConfig,
    rng: &mut rand::rngs::StdRng,
    rollout: &mut P,
    monitor: &Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: &Option<Arc<AtomicBool>>,
    buffer: &mut ReplayBuffer,
) -> (usize, Vec<f64>, usize) {
    let budget = Duration::from_secs_f64(config.self_play_secs.max(0.0));
    let phase_start = Instant::now();
    let mut new_records = 0usize;
    let mut game_secs: Vec<f64> = Vec::new();
    let mut game_i = 0usize;

    while phase_start.elapsed() < budget {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }

        game_i += 1;
        let t0 = Instant::now();
        let n_prog = config.self_play_progress_every_n_moves;
        let records = collector.play_game_with_progress(
            config.mcts_iters_per_move,
            rng,
            rollout,
            |move_idx, mcts_dt| {
                if let Some(m) = monitor {
                    if let Ok(mut g) = m.lock() {
                        g.current_game = game_i;
                        g.current_move = move_idx;
                        g.last_mcts_secs = mcts_dt.as_secs_f64();
                    }
                }
                if n_prog > 1 && (move_idx - 1) % n_prog != 0 {
                    return;
                }
                log_line(
                    monitor,
                    log_stdout,
                    &format!(
                        "    game {}  move {:4}  mcts {:6.2}s",
                        game_i,
                        move_idx,
                        mcts_dt.as_secs_f64()
                    ),
                );
            },
        );
        let secs = t0.elapsed().as_secs_f64();
        game_secs.push(secs);
        let positions = records.len();
        new_records += positions;
        for r in records {
            buffer.push(r);
        }
        if let Some(m) = monitor {
            if let Ok(mut g) = m.lock() {
                g.buffer_len = buffer.len();
            }
        }
        log_line(
            monitor,
            log_stdout,
            &format!(
                "  game {} done: {:.2}s ({} positions)",
                game_i, secs, positions
            ),
        );
    }

    (new_records, game_secs, game_i)
}

fn resolve_initial_checkpoint_path<'a>(
    latest: &'a str,
    best: &'a str,
    legacy: &'a str,
) -> Option<&'a str> {
    if Path::new(latest).is_file() {
        Some(latest)
    } else if Path::new(best).is_file() {
        Some(best)
    } else if Path::new(legacy).is_file() {
        Some(legacy)
    } else {
        None
    }
}

/// Prefer [`DEFAULT_LATEST_PATH`], then [`DEFAULT_BEST_PATH`], then [`DEFAULT_LEGACY_MODEL_PATH`].
/// Use this from the play UI so it loads the same weights as training when present.
pub fn default_inference_checkpoint_path() -> Option<&'static str> {
    resolve_initial_checkpoint_path(
        DEFAULT_LATEST_PATH,
        DEFAULT_BEST_PATH,
        DEFAULT_LEGACY_MODEL_PATH,
    )
}

/// Run new-vs-best games for `promotion_eval_secs`. Returns `(new_wins, games_played)`.
fn promotion_eval_games(
    collector: &SelfPlayCollector,
    config: &TrainingConfig,
    rng: &mut rand::rngs::StdRng,
    new_net: &HextoeNet,
    best_net: &HextoeNet,
    device: &Device,
    monitor: &Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: &Option<Arc<AtomicBool>>,
) -> (u32, u32) {
    let budget = Duration::from_secs_f64(config.promotion_eval_secs.max(0.0));
    let t0 = Instant::now();
    let mut new_wins = 0u32;
    let mut games = 0u32;

    while t0.elapsed() < budget {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }

        let new_player = if games % 2 == 0 {
            Player::X
        } else {
            Player::O
        };
        let w = collector.play_match_game(
            config.mcts_iters_per_move,
            rng,
            new_net,
            best_net,
            new_player,
            device,
        );
        games += 1;
        if let Some(winner) = w {
            if winner == new_player {
                new_wins += 1;
            }
        }
        let outcome = match w {
            Some(p) => format!("{p:?}"),
            None => "draw".to_string(),
        };
        log_line(
            monitor,
            log_stdout,
            &format!("    promotion game {games}: new as {new_player:?} → {outcome}"),
        );
    }

    (new_wins, games)
}

fn io_to_candle(e: std::io::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{e}"))
}

/// Run the training loop until [`cancel`] is set (if provided) or forever.
pub fn run_training(
    config: TrainingConfig,
    monitor: Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: Option<Arc<AtomicBool>>,
) -> candle_core::Result<()> {
    let device = config.device.clone();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = HextoeNet::new(vb)?;

    if let Some(path) = resolve_initial_checkpoint_path(
        &config.latest_path,
        &config.best_path,
        DEFAULT_LEGACY_MODEL_PATH,
    ) {
        let mut vm = varmap.clone();
        match load_weights(&mut vm, path) {
            Ok(()) => log_line(
                &monitor,
                log_stdout,
                &format!("Loaded weights from {path}"),
            ),
            Err(e) => log_line(
                &monitor,
                log_stdout,
                &format!("Could not load weights ({e}); starting fresh"),
            ),
        }
    } else {
        log_line(&monitor, log_stdout, "No checkpoint found — starting with random weights");
    }

    let adam_params = ParamsAdamW {
        lr: config.lr,
        weight_decay: config.weight_decay,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adam_params)?;
    let mut buffer = ReplayBuffer::new(config.replay_capacity);
    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::from_entropy();

    let rollout_note = if config.use_random_rollout {
        "random playouts"
    } else {
        "NN policy rollouts"
    };
    log_line(
        &monitor,
        log_stdout,
        &format!(
            "Training loop (~{:.0}s self-play / iter, {} MCTS iters/move, {rollout_note}). latest={} best={}",
            config.self_play_secs,
            config.mcts_iters_per_move,
            config.latest_path,
            config.best_path,
        ),
    );

    let mut iteration = 0u32;
    loop {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            log_line(&monitor, log_stdout, "Training stopped by user.");
            break;
        }

        iteration += 1;

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.iteration = iteration;
                g.phase = TrainPhase::SelfPlay;
                g.buffer_len = buffer.len();
            }
        }

        let sp_rollout_note = if config.use_random_rollout {
            "random playouts"
        } else {
            "NN policy rollouts"
        };
        log_line(
            &monitor,
            log_stdout,
            &format!(
                "[iter {iteration}] self-play ({sp_rollout_note}): {:.0}s budget × {} MCTS iters/move — starting…",
                config.self_play_secs,
                config.mcts_iters_per_move
            ),
        );

        let sp_phase = Instant::now();
        let (new_records, game_secs, games_played) = if config.use_random_rollout {
            let mut rollout = RandomRollout;
            self_play_until_duration(
                &collector,
                &config,
                &mut rng,
                &mut rollout,
                &monitor,
                log_stdout,
                &cancel,
                &mut buffer,
            )
        } else {
            let mut rollout = NeuralRollout {
                net: &model,
                device: &device,
            };
            self_play_until_duration(
                &collector,
                &config,
                &mut rng,
                &mut rollout,
                &monitor,
                log_stdout,
                &cancel,
                &mut buffer,
            )
        };

        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            log_line(&monitor, log_stdout, "Training stopped by user.");
            break;
        }

        let sp_total = sp_phase.elapsed().as_secs_f64();
        let sum_g: f64 = game_secs.iter().sum();
        let (avg_g, min_g, max_g) = if game_secs.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let n = game_secs.len() as f64;
            let min_g = game_secs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_g = game_secs.iter().copied().fold(0.0_f64, f64::max);
            (sum_g / n, min_g, max_g)
        };
        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.last_self_play_total_secs = sp_total;
                g.last_iteration_new_records = new_records;
                g.buffer_len = buffer.len();
                g.games_this_iter = games_played;
            }
        }
        log_line(
            &monitor,
            log_stdout,
            &format!(
                "[iter {iteration}] +{new_records} records  buffer {}/{}  |  self-play {sp_total:.1}s, {games_played} games ({avg_g:.2}s/game avg, {min_g:.2}s–{max_g:.2}s min–max)",
                buffer.len(),
                config.replay_capacity
            ),
        );

        if buffer.len() < config.min_buffer_for_training {
            log_line(
                &monitor,
                log_stdout,
                "  (buffer too small, skipping training)",
            );
            continue;
        }

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.phase = TrainPhase::Training;
            }
        }

        let mut total_loss = 0.0f32;
        for _ in 0..config.train_steps {
            let batch = buffer.sample_batch(config.batch_size, &mut rng);
            let loss = train_step(&model, &batch, &device, &mut opt)?;
            total_loss += loss;
        }
        let mean_loss = total_loss / config.train_steps as f32;
        log_line(
            &monitor,
            log_stdout,
            &format!("  mean loss = {mean_loss:.4}"),
        );

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.mean_loss = Some(mean_loss);
                g.phase = TrainPhase::SavingCheckpoint;
            }
        }

        let mut vm = varmap.clone();
        save_weights(&mut vm, &config.latest_path)?;
        let ck_msg = format!("checkpoint → {}", config.latest_path);
        log_line(&monitor, log_stdout, &format!("  {ck_msg}"));
        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.last_checkpoint_msg = Some(ck_msg);
            }
        }

        // Promotion gate: keep `best` unless the new net fails the eval (NN mode only).
        if config.use_random_rollout {
            if !Path::new(&config.best_path).is_file() {
                fs::copy(&config.latest_path, &config.best_path).map_err(io_to_candle)?;
                let msg = format!(
                    "promotion: copied latest → best (no prior best; random rollout mode)"
                );
                log_line(&monitor, log_stdout, &format!("  {msg}"));
                if let Some(m) = &monitor {
                    if let Ok(mut g) = m.lock() {
                        g.last_promotion_msg = Some(msg);
                        g.phase = TrainPhase::Idle;
                    }
                }
            } else {
                let msg = "promotion: skipped (--random-rollout)";
                log_line(&monitor, log_stdout, &format!("  {msg}"));
                if let Some(m) = &monitor {
                    if let Ok(mut g) = m.lock() {
                        g.last_promotion_msg = Some(msg.to_string());
                        g.phase = TrainPhase::Idle;
                    }
                }
            }
            continue;
        }

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.phase = TrainPhase::PromotionEval;
            }
        }

        if !Path::new(&config.best_path).is_file() {
            fs::copy(&config.latest_path, &config.best_path).map_err(io_to_candle)?;
            let msg = "promotion: first best checkpoint (copied from latest)".to_string();
            log_line(&monitor, log_stdout, &format!("  {msg}"));
            if let Some(m) = &monitor {
                if let Ok(mut g) = m.lock() {
                    g.last_promotion_msg = Some(msg);
                    g.phase = TrainPhase::Idle;
                }
            }
            continue;
        }

        let best_loaded = LoadedNet::try_load(&config.best_path, &device)?;
        log_line(
            &monitor,
            log_stdout,
            &format!(
                "  promotion eval: {:.0}s new vs best ({})…",
                config.promotion_eval_secs, config.best_path
            ),
        );
        let (new_wins, promo_games) = promotion_eval_games(
            &collector,
            &config,
            &mut rng,
            &model,
            &best_loaded.net,
            &device,
            &monitor,
            log_stdout,
            &cancel,
        );

        let rate = if promo_games > 0 {
            new_wins as f64 / promo_games as f64
        } else {
            0.0
        };
        let promoted = promo_games > 0 && rate >= config.promotion_min_win_rate;
        let msg = if promo_games == 0 {
            "promotion: no eval games in budget — best unchanged".to_string()
        } else if promoted {
            fs::copy(&config.latest_path, &config.best_path).map_err(io_to_candle)?;
            format!(
                "promotion: new wins {new_wins}/{promo_games} ({rate:.1}%) ≥ {:.0}% → updated best",
                config.promotion_min_win_rate * 100.0
            )
        } else {
            format!(
                "promotion: new wins {new_wins}/{promo_games} ({rate:.1}%) < {:.0}% — best unchanged",
                config.promotion_min_win_rate * 100.0
            )
        };
        log_line(&monitor, log_stdout, &format!("  {msg}"));
        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.last_promotion_msg = Some(msg);
                g.phase = TrainPhase::Idle;
            }
        }
    }

    Ok(())
}

pub fn train_step(
    model: &HextoeNet,
    batch: &[&crate::self_play::GameRecord],
    device: &Device,
    opt: &mut AdamW,
) -> candle_core::Result<f32> {
    let b = batch.len();

    let mut state_data = Vec::with_capacity(b * CHANNELS * GRID * GRID);
    let mut pi_data = Vec::with_capacity(b * GRID * GRID);
    let mut z_data = Vec::with_capacity(b);

    for rec in batch {
        state_data.extend_from_slice(rec.state_enc.as_ref());
        pi_data.extend_from_slice(rec.pi.as_ref());
        z_data.push(rec.outcome);
    }

    let states = Tensor::from_slice(
        &state_data,
        (b, CHANNELS, GRID, GRID),
        device,
    )?;
    let target_pi = Tensor::from_slice(&pi_data, (b, GRID * GRID), device)?;
    let target_z = Tensor::from_slice(&z_data, (b, 1usize), device)?;

    let (policy_logits, value) = model.forward(&states)?;

    let log_p = candle_nn::ops::log_softmax(&policy_logits, 1)?;
    let policy_loss = (&target_pi * &log_p)?.neg()?.mean_all()?;

    let value_loss = (&value - &target_z)?.sqr()?.mean_all()?;

    let loss = (&policy_loss + &value_loss)?;
    opt.backward_step(&loss)?;

    loss.to_scalar::<f32>()
}
