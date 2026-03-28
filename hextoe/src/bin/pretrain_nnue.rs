//! Supervised pre-training of the NNUE value network from online game data (JSON).
//!
//! Usage:
//!   hextoe-pretrain-nnue <games1.json> [games2.json ...] [--epochs N] [--batch-size B]
//!                        [--lr LR] [--out path] [--positions-per-epoch N]
//!
//! Accepts one or more JSON files (pass a shell glob and the shell will expand it).
//! Trains only the value head (MSE loss). The NNUE has no policy head.
//!
//! --positions-per-epoch N
//!   Each epoch trains on a random subset of N positions instead of the full dataset.
//!   Useful when the dataset is large: short epochs give frequent progress reports and
//!   checkpoints while still seeing the full dataset over many epochs.
//!   Default: use all positions every epoch.

use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hextoe::nnue::{build_nnue_model, NNUENet, DEFAULT_NNUE_PATH};
use hextoe::supervised::load_nnue_records_multi;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with('-') {
        eprintln!(
            "Usage: hextoe-pretrain-nnue <games1.json> [games2.json ...] \
             [--epochs N] [--batch-size B] [--lr LR] [--out path] [--positions-per-epoch N]"
        );
        std::process::exit(1);
    }

    let json_paths: Vec<String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with('-'))
        .cloned()
        .collect();
    let epochs = parse_arg(&args, "--epochs", 300usize);
    let batch_size = parse_arg(&args, "--batch-size", 256usize);
    let lr = parse_arg_f64(&args, "--lr", 1e-3);
    let out_path = parse_arg_str(&args, "--out", DEFAULT_NNUE_PATH);
    let positions_per_epoch = parse_arg(&args, "--positions-per-epoch", 0usize); // 0 = all

    println!("Loading game data from {} file(s)…", json_paths.len());
    let (records, used, skipped) = match load_nnue_records_multi(&json_paths) {
        Ok(x) => x,
        Err(e) => { eprintln!("Error loading games: {e}"); std::process::exit(1); }
    };
    println!("Total: {} games used, {} skipped → {} positions", used, skipped, records.len());
    if records.is_empty() {
        eprintln!("No records with NNUE features — nothing to train on.");
        std::process::exit(1);
    }

    let device = Device::Cpu;
    let (mut nnue_varmap, nnue_model) = build_nnue_model(&device).expect("build NNUE model");

    if Path::new(&out_path).exists() {
        match nnue_varmap.load(&out_path) {
            Ok(()) => println!("Loaded existing NNUE weights from {out_path}"),
            Err(e) => println!("Could not load {out_path}: {e} — training from scratch"),
        }
    } else {
        println!("No existing checkpoint — training from scratch");
    }

    let adam_params = ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() };
    let mut opt = AdamW::new(nnue_varmap.all_vars(), adam_params).expect("optimizer");

    let epoch_size = if positions_per_epoch > 0 {
        positions_per_epoch.min(records.len())
    } else {
        records.len()
    };
    let steps_per_epoch = (epoch_size + batch_size - 1) / batch_size;
    println!(
        "Training NNUE: {} epochs × {} steps/epoch (batch {}, {}/{} positions/epoch), lr={lr:.2e}",
        epochs, steps_per_epoch, batch_size, epoch_size, records.len()
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..records.len()).collect();
    let t0 = Instant::now();
    let mut best_loss = f32::MAX;
    let mut last_save = Instant::now();

    for epoch in 1..=epochs {
        let epoch_t0 = Instant::now();
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0usize;

        for chunk in indices[..epoch_size].chunks(batch_size) {
            let features_batch: Vec<Vec<usize>> = chunk
                .iter()
                .map(|&i| records[i].feats.iter().map(|&f| f as usize).collect())
                .collect();
            let z_data: Vec<f32> = chunk.iter().map(|&i| records[i].outcome).collect();
            let b = features_batch.len();

            let result = (|| -> candle_core::Result<f32> {
                let input = NNUENet::dense_from_sparse(&features_batch, &device)?;
                let target = Tensor::from_slice(&z_data, (b, 1usize), &device)?;
                let output = nnue_model.forward(&input)?;
                let loss = (&output - &target)?.sqr()?.mean_all()?;
                opt.backward_step(&loss)?;
                loss.to_scalar::<f32>()
            })();

            match result {
                Ok(loss) => { epoch_loss += loss; epoch_steps += 1; }
                Err(e) => eprintln!("train step error: {e}"),
            }
        }

        let mean_loss = if epoch_steps > 0 { epoch_loss / epoch_steps as f32 } else { 0.0 };
        let improved = mean_loss < best_loss;
        if improved { best_loss = mean_loss; }

        let elapsed = t0.elapsed().as_secs_f64();
        let secs_per_epoch = epoch_t0.elapsed().as_secs_f64();
        let eta_secs = secs_per_epoch * (epochs - epoch) as f64;
        let (eta_val, eta_unit) = if eta_secs >= 3600.0 {
            (eta_secs / 3600.0, "h")
        } else if eta_secs >= 60.0 {
            (eta_secs / 60.0, "m")
        } else {
            (eta_secs, "s")
        };
        let should_save = improved || last_save.elapsed().as_secs() >= 60;
        let save_marker = if should_save { "  [saved]" } else { "" };
        let marker = if improved { " ↓" } else { "  " };
        println!(
            "epoch {:4}/{epochs}  loss {mean_loss:.5}{marker}  best {best_loss:.5}  \
             {secs_per_epoch:.1}s/ep  ETA {eta_val:.1}{eta_unit}  (total {elapsed:.0}s){save_marker}",
            epoch,
        );
        if should_save {
            if let Err(e) = nnue_varmap.save(&out_path) {
                eprintln!("Failed to save: {e}");
            }
            last_save = Instant::now();
        }
    }

    match nnue_varmap.save(&out_path) {
        Ok(()) => println!("Saved → {out_path}"),
        Err(e) => eprintln!("Failed to save: {e}"),
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
