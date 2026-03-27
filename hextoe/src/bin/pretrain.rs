//! Supervised pre-training from online game data (JSON).
//!
//! Usage:
//!   hextoe-pretrain <games.json> [--epochs N] [--batch-size B] [--lr LR] [--out path]
//!
//! Loads human game records, trains the dual-head network on them (value head learns
//! win/loss, policy head learns to predict human moves), and saves the result.
//! Run this once before self-play training to give the network a warm start.

use std::path::Path;
use std::time::Instant;

use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hextoe::device::default_inference_device;
use hextoe::nn::build_model;
use hextoe::self_play::GameRecord;
use hextoe::supervised::load_supervised_records;
use hextoe::train::{train_step, DEFAULT_BEST_PATH, DEFAULT_LATEST_PATH};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with('-') {
        eprintln!("Usage: hextoe-pretrain <games.json> [--epochs N] [--batch-size B] [--lr LR] [--out path]");
        std::process::exit(1);
    }

    let json_path = &args[1];
    let epochs = parse_arg(&args, "--epochs", 200usize);
    let batch_size = parse_arg(&args, "--batch-size", 128usize);
    let lr = parse_arg_f64(&args, "--lr", 3e-4);
    let out_path = parse_arg_str(&args, "--out", DEFAULT_BEST_PATH);

    println!("Loading game data from {json_path} ...");
    let (records, used, skipped) = match load_supervised_records(json_path) {
        Ok(x) => x,
        Err(e) => { eprintln!("Error loading games: {e}"); std::process::exit(1); }
    };
    println!(
        "  {} games used, {} skipped → {} positions",
        used, skipped, records.len()
    );
    if records.is_empty() {
        eprintln!("No usable records found.");
        std::process::exit(1);
    }

    let device = default_inference_device();
    println!("Device: {device:?}");

    let (mut varmap, model) = build_model(&device).expect("build model");

    // Load existing weights if present so we can fine-tune rather than train from scratch.
    let existing = if Path::new(DEFAULT_BEST_PATH).exists() {
        DEFAULT_BEST_PATH
    } else if Path::new(DEFAULT_LATEST_PATH).exists() {
        DEFAULT_LATEST_PATH
    } else {
        ""
    };
    if !existing.is_empty() {
        match varmap.load(existing) {
            Ok(()) => println!("Loaded existing weights from {existing}"),
            Err(e) => println!("Could not load {existing}: {e} — training from scratch"),
        }
    } else {
        println!("No existing checkpoint found — training from scratch");
    }

    let adam_params = ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), adam_params).expect("optimizer");

    let steps_per_epoch = (records.len() + batch_size - 1) / batch_size;
    println!(
        "Training: {} epochs × {} steps/epoch (batch {}), lr={lr:.2e}",
        epochs, steps_per_epoch, batch_size
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..records.len()).collect();
    let t0 = Instant::now();

    for epoch in 1..=epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0usize;

        for chunk in indices.chunks(batch_size) {
            let batch: Vec<&GameRecord> = chunk.iter().map(|&i| &records[i]).collect();
            match train_step(&model, &batch, &device, &mut opt) {
                Ok(loss) => {
                    epoch_loss += loss;
                    epoch_steps += 1;
                }
                Err(e) => eprintln!("train_step error: {e}"),
            }
        }

        if epoch % 10 == 0 || epoch == 1 || epoch == epochs {
            let mean_loss = if epoch_steps > 0 { epoch_loss / epoch_steps as f32 } else { 0.0 };
            let elapsed = t0.elapsed().as_secs_f64();
            println!("epoch {:4}/{epochs}  loss {mean_loss:.4}  ({elapsed:.1}s elapsed)", epoch);
        }
    }

    // Save to the requested path AND sync both latest/best so training can pick up from here.
    for path in [out_path.as_str(), DEFAULT_BEST_PATH, DEFAULT_LATEST_PATH] {
        match varmap.save(path) {
            Ok(()) => println!("Saved → {path}"),
            Err(e) => eprintln!("Failed to save {path}: {e}"),
        }
    }
    println!("Done in {:.1}s", t0.elapsed().as_secs_f64());
}

// ── CLI arg helpers ───────────────────────────────────────────────────────────

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn parse_arg_f64(args: &[String], flag: &str, default: f64) -> f64 {
    parse_arg(args, flag, default)
}

fn parse_arg_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}
