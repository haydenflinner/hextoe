//! Shared self-play training loop (AlphaZero-style) used by `hextoe-train` and `hextoe-train-gui`.

use candle_core::{Device, DType, Tensor};
use candle_nn::{optim::AdamW, optim::ParamsAdamW, Optimizer, VarBuilder, VarMap};
use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::device::default_inference_device;
use crate::encode::{CHANNELS, GRID};
use crate::game::Player;
use crate::mcts::{RandomRollout, RolloutPolicy};
use crate::nn::{build_model, load_weights, save_weights, HextoeNet, LoadedNet, NeuralRollout};
use crate::nnue::{build_nnue_model, NNUENet, NNUEWeights, NNUERollout, DEFAULT_NNUE_PATH};
use crate::self_play::{GameRecord, ReplayBuffer, SelfPlayCollector};

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
pub const DEFAULT_SELF_PLAY_SECS: f64 = 60.0 * 3.0;
/// Wall-clock budget for new-vs-best games after training.
pub const DEFAULT_PROMOTION_EVAL_SECS: f64 = 60.0;
/// Promote `latest` → `best` if new wins at least this fraction of eval games (no draws counted in denominator).
pub const DEFAULT_PROMOTION_MIN_WIN_RATE: f64 = 0.52;
pub const DEFAULT_MCTS_ITERS_PER_MOVE: u32 = 100;
pub const DEFAULT_TRAIN_STEPS: usize = 10;
pub const DEFAULT_BATCH_SIZE: usize = 128;
pub const DEFAULT_LR: f64 = 3e-4;
pub const DEFAULT_WEIGHT_DECAY: f64 = 1e-4;
pub const DEFAULT_SELF_PLAY_PROGRESS_EVERY_N_MOVES: u32 = 1;
/// Concurrent self-play games per batch (`0` = use [`rayon::current_num_threads`]).
pub const DEFAULT_SELF_PLAY_PARALLEL_GAMES: usize = 0;
/// Minimum number of NN-vs-NN games in the promotion gate (keeps playing past the time budget
/// until this many games have been played).
pub const DEFAULT_MIN_PROMOTION_GAMES: u32 = 20;
/// After this many consecutive promotion failures, unconditionally promote to avoid stagnation.
/// `0` disables forced promotion.
pub const DEFAULT_MAX_STAGNATION_ROUNDS: u32 = 5;
/// Population size for tournament-based training (`1` = classic single-model loop).
pub const DEFAULT_POPULATION_SIZE: usize = 1;

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
    /// If true, MCTS simulations use fast uniform random playouts instead of NN value at the leaf.
    pub use_random_rollout: bool,
    /// If true, use the NNUE value network for leaf evaluation (pure CPU, fully parallel).
    /// Takes precedence over `use_random_rollout` when both are set.
    pub use_nnue_rollout: bool,
    /// Path for the NNUE model checkpoint (saved/loaded alongside the CNN).
    pub nnue_path: String,
    /// How many games to run in parallel during each self-play batch (`0` = Rayon thread count).
    ///
    /// With NN leaf evaluation on a GPU, [`parallel_game_count`] is effectively capped at `1` so
    /// only one thread uses the device at a time (Candle/Metal is not safe for concurrent inference).
    pub self_play_parallel_games: usize,
    /// Minimum number of NN-vs-NN games for promotion (plays past time budget if needed).
    pub min_promotion_games: u32,
    /// After this many consecutive promotion failures, force-promote to escape stagnation.
    /// `0` disables forced promotion.
    pub max_stagnation_rounds: u32,
    /// Number of candidate models per iteration (`1` = classic loop; `>1` = population tournament).
    pub population_size: usize,
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
            device: default_inference_device(),
            use_random_rollout,
            use_nnue_rollout: false,
            nnue_path: DEFAULT_NNUE_PATH.to_string(),
            self_play_parallel_games: DEFAULT_SELF_PLAY_PARALLEL_GAMES,
            min_promotion_games: DEFAULT_MIN_PROMOTION_GAMES,
            max_stagnation_rounds: DEFAULT_MAX_STAGNATION_ROUNDS,
            population_size: DEFAULT_POPULATION_SIZE,
        }
    }

    /// Same as [`TrainingConfig::default`], but applies CLI overrides (`--random-rollout`,
    /// `--population N`). Auto-enables NNUE rollout if the NNUE checkpoint exists and
    /// neither `--nnue` nor `--random-rollout` was explicitly passed.
    pub fn default_with_cli_rollout() -> Self {
        let mut cfg = Self::from_defaults(cli_use_random_rollout());
        cfg.population_size = cli_population_size();
        if cli_use_nnue_rollout() {
            cfg.use_nnue_rollout = true;
            cfg.use_random_rollout = false;
        } else if !cli_use_random_rollout() && Path::new(&cfg.nnue_path).exists() {
            cfg.use_nnue_rollout = true;
        }
        cfg
    }
}

/// True if argv contains `--random-rollout` or `-r`.
pub fn cli_use_random_rollout() -> bool {
    std::env::args().skip(1).any(|a| a == "--random-rollout" || a == "-r")
}

/// True if argv contains `--nnue` (use NNUE leaf evaluation for self-play).
pub fn cli_use_nnue_rollout() -> bool {
    std::env::args().skip(1).any(|a| a == "--nnue")
}

/// True if argv contains `--one-checkpoint` (for `hextoe-train` profiling runs).
pub fn cli_one_checkpoint() -> bool {
    std::env::args().skip(1).any(|a| a == "--one-checkpoint")
}

/// Parse `--population N` / `-p N` from argv (`1` if absent).
pub fn cli_population_size() -> usize {
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate() {
        if (a == "--population" || a == "-p") && i + 1 < args.len() {
            if let Ok(n) = args[i + 1].parse::<usize>() {
                return n.max(1);
            }
        }
    }
    DEFAULT_POPULATION_SIZE
}

/// Effective number of concurrent self-play games (`0` in config → Rayon thread count).
///
/// NN rollouts on a non-CPU [`Device`] are limited to one concurrent game: multiple Rayon workers
/// would call into the same Candle GPU context and can abort on Metal.
/// NNUE and random rollouts are pure CPU and thread-safe, so they get the full thread count.
pub fn parallel_game_count(config: &TrainingConfig) -> usize {
    let n = match config.self_play_parallel_games {
        0 => rayon::current_num_threads().max(1),
        n => n.max(1),
    };
    // NNUE is always pure CPU + Arc — fully parallel.
    if config.use_nnue_rollout {
        return n;
    }
    // Metal NN rollout: serial only.
    if !config.use_random_rollout && !matches!(config.device, Device::Cpu) {
        return 1;
    }
    n
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
    pub use_random_rollout: bool,
    pub use_nnue_rollout: bool,
    pub self_play_parallel_games: usize,
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
            use_nnue_rollout: config.use_nnue_rollout,
            self_play_parallel_games: parallel_game_count(config),
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

/// Play full games until `self_play_secs` wall time has elapsed (no new batch starts after the budget).
///
/// When [`parallel_game_count`] is more than one, each batch runs that many games on Rayon (shared
/// read-only `rollout`; each game uses its own RNG). With NN rollouts on GPU, the effective count is
/// `1` — see [`parallel_game_count`].
pub fn self_play_until_duration<P: RolloutPolicy + Send + Sync>(
    collector: &SelfPlayCollector,
    config: &TrainingConfig,
    rng: &mut rand::rngs::StdRng,
    rollout: &P,
    monitor: &Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: &Option<Arc<AtomicBool>>,
    buffer: &mut ReplayBuffer,
) -> (usize, Vec<f64>, usize) {
    let budget = Duration::from_secs_f64(config.self_play_secs.max(0.0));
    let phase_start = Instant::now();
    let deadline = phase_start + budget;
    let parallel = parallel_game_count(config);

    if parallel <= 1 {
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

        return (new_records, game_secs, game_i);
    }

    let game_counter = AtomicUsize::new(0);
    let mut new_records = 0usize;
    let mut game_secs: Vec<f64> = Vec::new();

    loop {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }
        if phase_start.elapsed() >= budget {
            break;
        }

        let batch: Vec<(f64, usize, Vec<GameRecord>, usize)> = (0..parallel)
            .into_par_iter()
            .filter_map(|_| {
                if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
                    return None;
                }
                if Instant::now() >= deadline {
                    return None;
                }
                let game_i = game_counter.fetch_add(1, Ordering::Relaxed) + 1;
                let mut thread_rng = rand::thread_rng();
                let t0 = Instant::now();
                let n_prog = config.self_play_progress_every_n_moves;
                let records = collector.play_game_with_progress(
                    config.mcts_iters_per_move,
                    &mut thread_rng,
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
                let positions = records.len();
                Some((secs, positions, records, game_i))
            })
            .collect();

        if batch.is_empty() {
            break;
        }

        for (secs, positions, records, gid) in batch {
            game_secs.push(secs);
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
                    "  game {gid} done: {:.2}s ({} positions)",
                    secs, positions
                ),
            );
        }
    }

    let games_played = game_counter.load(Ordering::Relaxed);
    (new_records, game_secs, games_played)
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

/// Run new-vs-best games until at least `min_promotion_games` have been played and
/// `promotion_eval_secs` have elapsed. Returns `(new_wins, games_played)`.
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
    let min_games = config.min_promotion_games;
    // Promotion gating is only meant to pick a better model. Using the full self-play MCTS
    // budget here is unnecessarily expensive because it runs many NN leaf evaluations.
    let eval_mcts_iters = (config.mcts_iters_per_move / 4).max(10);
    let t0 = Instant::now();
    let mut new_wins = 0u32;
    let mut games = 0u32;

    while t0.elapsed() < budget || games < min_games {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }

        let new_player = if games % 2 == 0 {
            Player::X
        } else {
            Player::O
        };
        let w = collector.play_match_game(
            eval_mcts_iters,
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

/// Play a fixed number of NN-vs-NN games between two networks. Returns `(a_wins, b_wins)`.
fn head_to_head(
    collector: &SelfPlayCollector,
    mcts_iters: u32,
    rng: &mut rand::rngs::StdRng,
    net_a: &HextoeNet,
    net_b: &HextoeNet,
    device: &Device,
    games: u32,
) -> (u32, u32) {
    let mut a_wins = 0u32;
    let mut b_wins = 0u32;
    for g in 0..games {
        let a_player = if g % 2 == 0 { Player::X } else { Player::O };
        let winner = collector.play_match_game(mcts_iters, rng, net_a, net_b, a_player, device);
        match winner {
            Some(p) if p == a_player => a_wins += 1,
            Some(_) => b_wins += 1,
            None => {}
        }
    }
    (a_wins, b_wins)
}

/// Round-robin tournament among `nets`. Returns win counts for each participant.
fn round_robin_tournament(
    collector: &SelfPlayCollector,
    mcts_iters: u32,
    rng: &mut rand::rngs::StdRng,
    nets: &[&HextoeNet],
    device: &Device,
    games_per_pair: u32,
    monitor: &Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
) -> Vec<u32> {
    let n = nets.len();
    let mut wins = vec![0u32; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (wi, wj) = head_to_head(
                collector,
                mcts_iters,
                rng,
                nets[i],
                nets[j],
                device,
                games_per_pair,
            );
            wins[i] += wi;
            wins[j] += wj;
            log_line(
                monitor,
                log_stdout,
                &format!("    candidate {i} vs {j}: {wi}-{wj}"),
            );
        }
    }
    wins
}

/// Population-based training loop. Each iteration trains N candidate models with randomized
/// hyperparameters, then runs a round-robin tournament (including the incumbent best) and
/// keeps the winner.
fn run_population_training(
    config: TrainingConfig,
    monitor: Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: Option<Arc<AtomicBool>>,
) -> candle_core::Result<()> {
    let device = config.device.clone();
    let pop_size = config.population_size.max(2);
    let games_per_pair: u32 = 2;
    let eval_mcts_iters = (config.mcts_iters_per_move / 4).max(10);
    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut buffer = ReplayBuffer::new(config.replay_capacity);

    // Seed the best checkpoint from existing weights if available.
    if !Path::new(&config.best_path).is_file() {
        if let Some(src) = resolve_initial_checkpoint_path(
            &config.latest_path,
            &config.best_path,
            DEFAULT_LEGACY_MODEL_PATH,
        ) {
            fs::copy(src, &config.best_path).map_err(io_to_candle)?;
            log_line(
                &monitor,
                log_stdout,
                &format!("population: seeded best from {src}"),
            );
        } else {
            let (vm, _) = build_model(&device)?;
            save_weights(&vm, &config.best_path)?;
            log_line(
                &monitor,
                log_stdout,
                "population: no checkpoint found — initialised random best",
            );
        }
    }

    log_line(
        &monitor,
        log_stdout,
        &format!(
            "Population training: {} candidates + incumbent, {:.0}s self-play, {} MCTS iters/move (self-play), {} MCTS iters/move (eval), {} games/pair",
            pop_size, config.self_play_secs, config.mcts_iters_per_move, eval_mcts_iters, games_per_pair
        ),
    );

    let mut iteration = 0u32;
    loop {
        if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            log_line(&monitor, log_stdout, "Population training stopped by user.");
            break;
        }

        iteration += 1;
        log_line(&monitor, log_stdout, &format!("[pop iter {iteration}] starting…"));

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.iteration = iteration;
                g.phase = TrainPhase::SelfPlay;
                g.buffer_len = buffer.len();
            }
        }

        // ── Self-play with random rollout ──
        let rollout = RandomRollout;
        let (new_records, game_secs, games_played) = self_play_until_duration(
            &collector,
            &config,
            &mut rng,
            &rollout,
            &monitor,
            log_stdout,
            &cancel,
            &mut buffer,
        );

        let sp_total: f64 = game_secs.iter().sum();
        log_line(
            &monitor,
            log_stdout,
            &format!(
                "[pop iter {iteration}] +{new_records} records  buffer {}/{}  |  {games_played} games in {sp_total:.1}s",
                buffer.len(),
                config.replay_capacity,
            ),
        );

        if buffer.len() < config.min_buffer_for_training {
            log_line(&monitor, log_stdout, "  (buffer too small, skipping training)");
            continue;
        }

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.phase = TrainPhase::Training;
            }
        }

        // ── Train N candidates from the current best with randomized hyperparameters ──
        let mut candidates: Vec<(VarMap, HextoeNet, String)> = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let (mut vm, net) = build_model(&device)?;
            load_weights(&mut vm, &config.best_path)?;

            let lr_factor: f64 = 10f64.powf(rng.gen_range(-0.5..0.5));
            let lr = config.lr * lr_factor;
            let steps_factor: f64 = rng.gen_range(0.5..2.0);
            let steps = ((config.train_steps as f64) * steps_factor).round().max(1.0) as usize;
            let wd_factor: f64 = 10f64.powf(rng.gen_range(-0.5..0.5));
            let wd = config.weight_decay * wd_factor;

            let adam_params = ParamsAdamW {
                lr,
                weight_decay: wd,
                ..Default::default()
            };
            let mut opt = AdamW::new(vm.all_vars(), adam_params)?;

            let mut total_loss = 0.0f32;
            for _ in 0..steps {
                let batch = buffer.sample_batch(config.batch_size, &mut rng);
                let loss = train_step(&net, &batch, &device, &mut opt)?;
                total_loss += loss;
            }
            let mean_loss = total_loss / steps as f32;
            let desc = format!(
                "cand {i}: lr={lr:.2e} wd={wd:.2e} steps={steps} loss={mean_loss:.4}"
            );
            log_line(&monitor, log_stdout, &format!("  {desc}"));
            candidates.push((vm, net, desc));
        }

        // ── Load incumbent (current best, untrained) ──
        let incumbent = LoadedNet::try_load(&config.best_path, &device)?;

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.phase = TrainPhase::PromotionEval;
            }
        }

        // ── Round-robin tournament: candidates + incumbent ──
        let mut all_nets: Vec<&HextoeNet> = candidates.iter().map(|(_, n, _)| n).collect();
        all_nets.push(&incumbent.net);
        let incumbent_idx = all_nets.len() - 1;

        log_line(&monitor, log_stdout, "  round-robin tournament…");
        let wins = round_robin_tournament(
            &collector,
            eval_mcts_iters,
            &mut rng,
            &all_nets,
            &device,
            games_per_pair,
            &monitor,
            log_stdout,
        );

        let (best_idx, best_wins) = wins
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| *w)
            .unwrap();

        let msg = if best_idx == incumbent_idx {
            format!(
                "population: incumbent wins ({best_wins} wins) — best unchanged"
            )
        } else {
            save_weights(&candidates[best_idx].0, &config.best_path)?;
            save_weights(&candidates[best_idx].0, &config.latest_path)?;
            format!(
                "population: {} wins ({best_wins} wins) → updated best",
                candidates[best_idx].2,
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

fn io_to_candle(e: std::io::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{e}"))
}

/// Run the training loop until [`cancel`] is set (if provided) or forever.
///
/// If `stop_after_first_checkpoint` is true, exits right after the first successful write to
/// [`TrainingConfig::latest_path`] (skips promotion eval). Useful with `cargo flamegraph`.
///
/// When [`TrainingConfig::population_size`] > 1, delegates to a population-based tournament
/// loop instead of the classic single-model AlphaZero loop.
pub fn run_training(
    config: TrainingConfig,
    monitor: Option<Arc<Mutex<TrainingMonitor>>>,
    log_stdout: bool,
    cancel: Option<Arc<AtomicBool>>,
    stop_after_first_checkpoint: bool,
) -> candle_core::Result<()> {
    if config.population_size > 1 {
        return run_population_training(config, monitor, log_stdout, cancel);
    }

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
    let mut one_checkpoint_buffer_hinted = false;
    let mut consecutive_failures = 0u32;

    // ── NNUE model (always built; trained alongside CNN; used for rollout when --nnue) ──
    let (mut nnue_varmap, nnue_model) = {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let net = NNUENet::new(vb).expect("NNUE build");
        (vm, net)
    };
    if Path::new(&config.nnue_path).is_file() {
        match nnue_varmap.load(&config.nnue_path) {
            Ok(()) => log_line(&monitor, log_stdout, &format!("NNUE: loaded from {}", config.nnue_path)),
            Err(e) => log_line(&monitor, log_stdout, &format!("NNUE: could not load ({}), starting fresh", e)),
        }
    } else {
        log_line(&monitor, log_stdout, "NNUE: no checkpoint found, starting with random weights");
    }
    let nnue_adam = ParamsAdamW { lr: config.lr, weight_decay: config.weight_decay, ..Default::default() };
    let mut nnue_opt = AdamW::new(nnue_varmap.all_vars(), nnue_adam)?;

    let rollout_note = if config.use_nnue_rollout {
        "NNUE leaf value (parallel)"
    } else if config.use_random_rollout {
        "random playouts"
    } else {
        "CNN leaf value"
    };
    log_line(
        &monitor,
        log_stdout,
        &format!(
            "Training loop ({:?}; ~{:.0}s self-play / iter, {} parallel games, {} MCTS iters/move, {rollout_note}). latest={} best={}",
            config.device,
            config.self_play_secs,
            parallel_game_count(&config),
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
                g.self_play_parallel_games = parallel_game_count(&config);
            }
        }

        let sp_rollout_note = if config.use_nnue_rollout {
            "NNUE leaf value"
        } else if config.use_random_rollout {
            "random playouts"
        } else {
            "CNN leaf value"
        };
        log_line(
            &monitor,
            log_stdout,
            &format!(
                "[iter {iteration}] self-play ({sp_rollout_note}): {:.0}s budget × {} parallel × {} MCTS iters/move — starting…",
                config.self_play_secs,
                parallel_game_count(&config),
                config.mcts_iters_per_move
            ),
        );

        let sp_phase = Instant::now();
        let (new_records, game_secs, games_played) = if config.use_nnue_rollout {
            // Extract current NNUE weights into a thread-safe Arc for parallel games.
            let weights = NNUEWeights::from_varmap(&nnue_varmap)
                .map(std::sync::Arc::new)
                .unwrap_or_else(|e| {
                    log_line(&monitor, log_stdout, &format!("  NNUE weight extraction failed: {e}; falling back to random rollout"));
                    // Fallback: build default weights (random; better than crashing)
                    let (vm, _) = build_nnue_model(&Device::Cpu).expect("nnue fallback");
                    std::sync::Arc::new(NNUEWeights::from_varmap(&vm).expect("nnue fallback weights"))
                });
            let rollout = NNUERollout::new(weights);
            self_play_until_duration(
                &collector,
                &config,
                &mut rng,
                &rollout,
                &monitor,
                log_stdout,
                &cancel,
                &mut buffer,
            )
        } else if config.use_random_rollout {
            let rollout = RandomRollout;
            self_play_until_duration(
                &collector,
                &config,
                &mut rng,
                &rollout,
                &monitor,
                log_stdout,
                &cancel,
                &mut buffer,
            )
        } else {
            let rollout = NeuralRollout {
                net: &model,
                device: &device,
            };
            self_play_until_duration(
                &collector,
                &config,
                &mut rng,
                &rollout,
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
            if stop_after_first_checkpoint && !one_checkpoint_buffer_hinted {
                one_checkpoint_buffer_hinted = true;
                log_line(
                    &monitor,
                    log_stdout,
                    "  --one-checkpoint: no checkpoint written yet; lower min_buffer_for_training or collect more self-play first.",
                );
            }
            continue;
        }

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.phase = TrainPhase::Training;
            }
        }

        let mut total_loss = 0.0f32;
        let mut nnue_total_loss = 0.0f32;
        let mut nnue_steps = 0usize;
        for _ in 0..config.train_steps {
            let batch = buffer.sample_batch(config.batch_size, &mut rng);
            let loss = train_step(&model, &batch, &device, &mut opt)?;
            total_loss += loss;
            if let Ok(Some(nl)) = nnue_train_step(&nnue_model, &batch, &Device::Cpu, &mut nnue_opt) {
                nnue_total_loss += nl;
                nnue_steps += 1;
            }
        }
        let mean_loss = total_loss / config.train_steps as f32;
        let nnue_loss_str = if nnue_steps > 0 {
            format!("  nnue loss = {:.4}", nnue_total_loss / nnue_steps as f32)
        } else {
            "  nnue loss = n/a (no NNUE-encoded records yet)".to_string()
        };
        log_line(
            &monitor,
            log_stdout,
            &format!("  cnn loss = {mean_loss:.4}  |  {nnue_loss_str}"),
        );

        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.mean_loss = Some(mean_loss);
                g.phase = TrainPhase::SavingCheckpoint;
            }
        }

        let mut vm = varmap.clone();
        save_weights(&mut vm, &config.latest_path)?;
        // Save NNUE checkpoint alongside CNN.
        if let Err(e) = nnue_varmap.save(&config.nnue_path) {
            log_line(&monitor, log_stdout, &format!("  NNUE save failed: {e}"));
        }
        let ck_msg = format!("checkpoint → {}  nnue → {}", config.latest_path, config.nnue_path);
        log_line(&monitor, log_stdout, &format!("  {ck_msg}"));
        if let Some(m) = &monitor {
            if let Ok(mut g) = m.lock() {
                g.last_checkpoint_msg = Some(ck_msg);
            }
        }

        if stop_after_first_checkpoint {
            log_line(
                &monitor,
                log_stdout,
                "  Stopping after first checkpoint (--one-checkpoint).",
            );
            break;
        }

        // Promotion gate: NN-vs-NN evaluation regardless of self-play rollout mode.
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
        let mut promoted = promo_games > 0 && rate >= config.promotion_min_win_rate;

        let force_threshold = config.max_stagnation_rounds;
        if !promoted && force_threshold > 0 {
            consecutive_failures += 1;
        }
        let force_promoted = !promoted
            && force_threshold > 0
            && consecutive_failures >= force_threshold;
        if force_promoted {
            promoted = true;
        }

        let msg = if promo_games == 0 {
            "promotion: no eval games in budget — best unchanged".to_string()
        } else if force_promoted {
            fs::copy(&config.latest_path, &config.best_path).map_err(io_to_candle)?;
            consecutive_failures = 0;
            format!(
                "promotion: new wins {new_wins}/{promo_games} ({:.1}%) — FORCE-PROMOTED after {} consecutive failures",
                rate * 100.0,
                force_threshold,
            )
        } else if promoted {
            fs::copy(&config.latest_path, &config.best_path).map_err(io_to_candle)?;
            consecutive_failures = 0;
            format!(
                "promotion: new wins {new_wins}/{promo_games} ({:.1}%) ≥ {:.0}% → updated best",
                rate * 100.0,
                config.promotion_min_win_rate * 100.0,
            )
        } else {
            format!(
                "promotion: new wins {new_wins}/{promo_games} ({:.1}%) < {:.0}% — best unchanged (stagnation {consecutive_failures}/{})",
                rate * 100.0,
                config.promotion_min_win_rate * 100.0,
                force_threshold,
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

/// Training step for the NNUE value network (MSE loss on value head only).
/// Records without NNUE features (`nnue_feats` empty) are silently skipped.
pub fn nnue_train_step(
    net: &NNUENet,
    batch: &[&GameRecord],
    _device: &Device,
    opt: &mut AdamW,
) -> candle_core::Result<Option<f32>> {
    // Filter to records that have NNUE features.
    let usable: Vec<&&GameRecord> = batch.iter().filter(|r| !r.nnue_feats.is_empty()).collect();
    if usable.is_empty() {
        return Ok(None);
    }
    let features_batch: Vec<Vec<usize>> = usable
        .iter()
        .map(|r| r.nnue_feats.iter().map(|&f| f as usize).collect())
        .collect();
    let z_data: Vec<f32> = usable.iter().map(|r| r.outcome).collect();
    let b = usable.len();

    // Always train NNUE on CPU regardless of the CNN device.
    let cpu = Device::Cpu;
    let target = Tensor::from_slice(&z_data, (b, 1usize), &cpu)?;

    let input = NNUENet::dense_from_sparse(&features_batch, &cpu)?;
    let output = net.forward(&input)?;
    let loss = (&output - &target)?.sqr()?.mean_all()?;
    opt.backward_step(&loss)?;

    Ok(Some(loss.to_scalar::<f32>()?))
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
