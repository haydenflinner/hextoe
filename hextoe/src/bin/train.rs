//! hextoe-train — self-play reinforcement learning loop (AlphaZero style).
//!
//! Runs indefinitely:
//!   1. Generate GAMES_PER_ITER self-play games using pure MCTS.
//!   2. Fill a replay buffer with (state, π, z) training records.
//!   3. Once the buffer is large enough, run TRAIN_STEPS gradient-descent steps.
//!   4. Save a checkpoint.
//!
//! The saved model can later be loaded by the GUI for NN-guided play.
//!
//! Usage:  cargo run --release --bin hextoe-train

use candle_core::{Device, DType, Tensor};
use candle_nn::{optim::AdamW, optim::ParamsAdamW, Optimizer, VarBuilder, VarMap};
use hextoe::encode::{CHANNELS, GRID};
use hextoe::nn::{load_weights, save_weights, HextoeNet};
use hextoe::self_play::{ReplayBuffer, SelfPlayCollector};
use rand::SeedableRng;

const MODEL_PATH: &str = "hextoe_model.safetensors";
const REPLAY_CAPACITY: usize = 50_000;
const MIN_BUFFER_FOR_TRAINING: usize = 500;
const GAMES_PER_ITER: usize = 10;
const MCTS_ITERS_PER_MOVE: u32 = 200;
const TRAIN_STEPS: usize = 50;
const BATCH_SIZE: usize = 128;
const LR: f64 = 3e-4;
const WEIGHT_DECAY: f64 = 1e-4;

fn main() {
    if let Err(e) = run() {
        eprintln!("Fatal error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // Build or load model.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = HextoeNet::new(vb)?;

    if std::path::Path::new(MODEL_PATH).exists() {
        let mut vm = varmap.clone();
        match load_weights(&mut vm, MODEL_PATH) {
            Ok(()) => println!("Loaded weights from {MODEL_PATH}"),
            Err(e) => println!("Could not load weights ({e}); starting fresh"),
        }
    } else {
        println!("No checkpoint found — starting with random weights");
    }

    let adam_params = ParamsAdamW {
        lr: LR,
        weight_decay: WEIGHT_DECAY,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adam_params)?;
    let mut buffer = ReplayBuffer::new(REPLAY_CAPACITY);
    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::from_entropy();

    let mut iteration = 0u32;
    loop {
        iteration += 1;

        // ── Self-play ──────────────────────────────────────────────────────
        let mut new_records = 0usize;
        for _ in 0..GAMES_PER_ITER {
            let records = collector.play_game(MCTS_ITERS_PER_MOVE, &mut rng);
            new_records += records.len();
            for r in records {
                buffer.push(r);
            }
        }
        println!(
            "[iter {iteration}] +{new_records} records  buffer {}/{}",
            buffer.len(),
            REPLAY_CAPACITY
        );

        // ── Training ──────────────────────────────────────────────────────
        if buffer.len() < MIN_BUFFER_FOR_TRAINING {
            println!("  (buffer too small, skipping training)");
            continue;
        }

        let mut total_loss = 0.0f32;
        for _ in 0..TRAIN_STEPS {
            let batch = buffer.sample_batch(BATCH_SIZE, &mut rng);
            let loss = train_step(&model, &batch, &device, &mut opt)?;
            total_loss += loss;
        }
        let mean_loss = total_loss / TRAIN_STEPS as f32;
        println!("  mean loss = {mean_loss:.4}");

        let mut vm = varmap.clone();
        save_weights(&mut vm, MODEL_PATH)?;
        println!("  checkpoint → {MODEL_PATH}");
    }
}

// ── Training step ─────────────────────────────────────────────────────────────

fn train_step(
    model: &HextoeNet,
    batch: &[&hextoe::self_play::GameRecord],
    device: &Device,
    opt: &mut AdamW,
) -> candle_core::Result<f32> {
    let b = batch.len();

    // Assemble flat slices.
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

    // Forward pass.
    let (policy_logits, value) = model.forward(&states)?;

    // Policy loss: cross-entropy  –Σ π·log p  (mean over batch)
    let log_p = candle_nn::ops::log_softmax(&policy_logits, 1)?;
    let policy_loss = (&target_pi * &log_p)?.neg()?.mean_all()?;

    // Value loss: MSE
    let value_loss = (&value - &target_z)?.sqr()?.mean_all()?;

    let loss = (&policy_loss + &value_loss)?;
    opt.backward_step(&loss)?;

    loss.to_scalar::<f32>()
}
