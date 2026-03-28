//! Supervised pre-training from online game data (JSON).
//!
//! Usage:
//!   hextoe-pretrain <games1.json> [games2.json ...] [--epochs N] [--batch-size B] [--lr LR] [--out path] [--cpu]
//!
//! Loads compact raw move lists, builds a sample index, then encodes each batch on the fly
//! with a random D₆ transform. Trains the dual-head CNN (value + policy) and saves every epoch.
//! Press CTRL+C once to save and exit cleanly; press again to force-quit.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
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
        eprintln!("Usage: hextoe-pretrain <games1.json> [games2.json ...] [--epochs N] [--batch-size B] [--lr LR] [--out path] [--cpu]");
        std::process::exit(1);
    }

    let json_paths: Vec<String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with('-'))
        .cloned()
        .collect();
    let epochs = parse_arg(&args, "--epochs", 200usize);
    let batch_size = parse_arg(&args, "--batch-size", 128usize);
    let lr = parse_arg_f64(&args, "--lr", 3e-4);
    let out_path = parse_arg_str(&args, "--out", DEFAULT_BEST_PATH);
    let force_cpu = args.iter().any(|a| a == "--cpu");

    // CTRL+C → clean save. Second CTRL+C → force-quit.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    ctrlc::set_handler(move || {
        if stop_clone.load(Ordering::Relaxed) {
            eprintln!("\nForce-quitting.");
            std::process::exit(1);
        }
        eprintln!("\nCTRL+C — saving checkpoint after this batch then exiting. Press again to force-quit.");
        stop_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting CTRL+C handler");

    println!("Loading game data from {} file(s)…", json_paths.len());
    let (games, used, skipped) = match load_raw_games_multi(&json_paths) {
        Ok(x) => x,
        Err(e) => { eprintln!("Error loading games: {e}"); std::process::exit(1); }
    };
    let sample_pairs = build_sample_index(&games);
    println!(
        "Total: {} games used, {} skipped → {} trainable positions (symmetry applied on-the-fly)",
        used, skipped, sample_pairs.len()
    );
    if sample_pairs.is_empty() {
        eprintln!("No usable samples found.");
        std::process::exit(1);
    }

    let device = if force_cpu { Device::Cpu } else { default_inference_device() };
    println!("Device: {device:?}{}", if force_cpu { " (--cpu)" } else { "" });

    let (mut varmap, model) = build_model(&device).expect("build model");

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
    let total_steps = epochs * steps_per_epoch;
    println!(
        "Training: {epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps (batch {batch_size}), lr={lr:.2e}",
    );
    println!("Checkpointing every epoch to {DEFAULT_LATEST_PATH}  |  CTRL+C to save and exit early");
    println!("{}", "─".repeat(80));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..sample_pairs.len()).collect();
    let t0 = Instant::now();
    let progress_every = (steps_per_epoch / 20).max(1); // ~5% increments

    'outer: for epoch in 1..=epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0usize;
        let epoch_t0 = Instant::now();

        for (chunk_i, chunk) in indices.chunks(batch_size).enumerate() {
            if stop.load(Ordering::Relaxed) {
                break 'outer;
            }

            let mut batch_owned: Vec<GameRecord> = Vec::with_capacity(chunk.len());
            for &pi in chunk {
                let (gi, si) = sample_pairs[pi];
                let tid = rng.gen_range(0u8..12);
                if let Some(rec) = encode_sample(&games, gi, si, tid) {
                    batch_owned.push(rec);
                }
            }
            if batch_owned.is_empty() { continue; }

            let batch_refs: Vec<&GameRecord> = batch_owned.iter().collect();
            match train_step(&model, &batch_refs, &device, &mut opt) {
                Ok(loss) => { epoch_loss += loss; epoch_steps += 1; }
                Err(e) => eprintln!("train_step error: {e}"),
            }

            // Per-step progress at ~5% intervals.
            if (chunk_i + 1) % progress_every == 0 || chunk_i + 1 == steps_per_epoch {
                let steps_done_total = (epoch - 1) * steps_per_epoch + chunk_i + 1;
                let elapsed = t0.elapsed().as_secs_f64();
                let secs_per_step = elapsed / steps_done_total as f64;
                let steps_left = total_steps.saturating_sub(steps_done_total);
                let eta_secs = secs_per_step * steps_left as f64;
                let mean_loss = if epoch_steps > 0 { epoch_loss / epoch_steps as f32 } else { 0.0 };
                println!(
                    "epoch {:>4}/{epochs}  step {:>6}/{steps_per_epoch}  loss {mean_loss:.4}  elapsed {:.0}s  ETA {:.0}s",
                    epoch, chunk_i + 1, elapsed, eta_secs
                );
            }
        }

        // Save checkpoint every epoch.
        let epoch_secs = epoch_t0.elapsed().as_secs_f64();
        let mean_loss = if epoch_steps > 0 { epoch_loss / epoch_steps as f32 } else { 0.0 };
        match varmap.save(DEFAULT_LATEST_PATH) {
            Ok(()) => println!(
                "── epoch {epoch}/{epochs} done  loss {mean_loss:.4}  ({epoch_secs:.1}s)  → saved {DEFAULT_LATEST_PATH}"
            ),
            Err(e) => eprintln!("── epoch {epoch}/{epochs} done  loss {mean_loss:.4}  ({epoch_secs:.1}s)  WARNING: save failed: {e}"),
        }
    }

    // Final save to all paths.
    println!("{}", "─".repeat(80));
    for path in [out_path.as_str(), DEFAULT_BEST_PATH, DEFAULT_LATEST_PATH] {
        match varmap.save(path) {
            Ok(()) => println!("Saved → {path}"),
            Err(e) => eprintln!("Failed to save {path}: {e}"),
        }
    }
    println!("Done in {:.1}s", t0.elapsed().as_secs_f64());
}

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
