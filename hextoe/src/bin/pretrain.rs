//! Supervised pre-training from online game data (JSON).
//!
//! Usage:
//!   hextoe-pretrain <games1.json> [games2.json ...] [--epochs N] [--batch-size B] [--lr LR] [--out path]
//!
//! Accepts one or more JSON files (pass a shell glob and the shell will expand it).
//! Loads compact raw move lists (not pre-expanded ×12 symmetries), builds a sample index,
//! then encodes each batch on the fly with a random D₆ transform. Trains the dual-head
//! network (value + policy) and saves the result.
//! Run this once before self-play training to give the network a warm start.

use std::path::Path;
use std::time::Instant;

use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use hextoe::device::default_inference_device;
use hextoe::nn::build_model;
use hextoe::self_play::GameRecord;
use hextoe::supervised::{build_sample_index, encode_sample, load_raw_games_multi};
use hextoe::train::{train_step, DEFAULT_BEST_PATH, DEFAULT_LATEST_PATH};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with('-') {
        eprintln!("Usage: hextoe-pretrain <games1.json> [games2.json ...] [--epochs N] [--batch-size B] [--lr LR] [--out path]");
        std::process::exit(1);
    }

    // Collect positional args (everything before the first --flag).
    let json_paths: Vec<String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with('-'))
        .cloned()
        .collect();
    let epochs = parse_arg(&args, "--epochs", 200usize);
    let batch_size = parse_arg(&args, "--batch-size", 128usize);
    let lr = parse_arg_f64(&args, "--lr", 3e-4);
    let out_path = parse_arg_str(&args, "--out", DEFAULT_BEST_PATH);

    println!("Loading game data from {} file(s)…", json_paths.len());
    let (games, used, skipped) = match load_raw_games_multi(&json_paths) {
        Ok(x) => x,
        Err(e) => { eprintln!("Error loading games: {e}"); std::process::exit(1); }
    };
    let sample_pairs = build_sample_index(&games);
    println!(
        "Total: {} games used, {} skipped → {} trainable (game, step) samples (symmetry on-the-fly)",
        used, skipped, sample_pairs.len()
    );
    if sample_pairs.is_empty() {
        eprintln!("No usable samples found.");
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

    let steps_per_epoch = (sample_pairs.len() + batch_size - 1) / batch_size;
    println!(
        "Training: {} epochs × {} steps/epoch (batch {}), lr={lr:.2e}",
        epochs, steps_per_epoch, batch_size
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..sample_pairs.len()).collect();
    let t0 = Instant::now();

    for epoch in 1..=epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0usize;

        for chunk in indices.chunks(batch_size) {
            let mut batch_owned: Vec<GameRecord> = Vec::with_capacity(chunk.len());
            for &pi in chunk {
                let (gi, si) = sample_pairs[pi];
                let tid = rng.gen_range(0u8..12);
                if let Some(rec) = encode_sample(&games, gi, si, tid) {
                    batch_owned.push(rec);
                }
            }
            if batch_owned.is_empty() {
                continue;
            }
            let batch_refs: Vec<&GameRecord> = batch_owned.iter().collect();
            match train_step(&model, &batch_refs, &device, &mut opt) {
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
