//! Supervised pre-training of the NNUE value network from online game data (JSON).
//!
//! Usage:
//!   hextoe-pretrain-nnue <games.json> [--epochs N] [--batch-size B] [--lr LR] [--out path]
//!
//! Trains only the value head (MSE loss). The NNUE has no policy head.
//! Typical workflow: run this once on human game data before self-play to give
//! the value network a warm start, then let the self-play loop keep it updated.

use std::path::Path;
use std::time::Instant;

use candle_core::Device;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hextoe::nnue::{build_nnue_model, DEFAULT_NNUE_PATH};
use hextoe::self_play::GameRecord;
use hextoe::supervised::load_supervised_records;
use hextoe::train::nnue_train_step;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with('-') {
        eprintln!(
            "Usage: hextoe-pretrain-nnue <games.json> [--epochs N] [--batch-size B] [--lr LR] [--out path]"
        );
        std::process::exit(1);
    }

    let json_path = &args[1];
    let epochs = parse_arg(&args, "--epochs", 300usize);
    let batch_size = parse_arg(&args, "--batch-size", 256usize);
    let lr = parse_arg_f64(&args, "--lr", 1e-3);
    let out_path = parse_arg_str(&args, "--out", DEFAULT_NNUE_PATH);

    println!("Loading game data from {json_path} …");
    let (records, used, skipped) = match load_supervised_records(json_path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            std::process::exit(1);
        }
    };
    println!("  {} games used, {} skipped → {} positions", used, skipped, records.len());

    // Check NNUE coverage.
    let nnue_count = records.iter().filter(|r| !r.nnue_feats.is_empty()).count();
    println!("  {} positions have NNUE features ({:.1}%)", nnue_count, 100.0 * nnue_count as f64 / records.len().max(1) as f64);
    if nnue_count == 0 {
        eprintln!("No records with NNUE features — nothing to train on.");
        std::process::exit(1);
    }

    // NNUE always trains on CPU.
    let device = Device::Cpu;
    let (mut nnue_varmap, nnue_model) = build_nnue_model(&device).expect("build NNUE model");

    // Load existing weights if present for fine-tuning.
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

    let steps_per_epoch = (nnue_count + batch_size - 1) / batch_size;
    println!(
        "Training NNUE: {} epochs × {} steps/epoch (batch {}), lr={lr:.2e}",
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
            match nnue_train_step(&nnue_model, &batch, &device, &mut opt) {
                Ok(Some(loss)) => {
                    epoch_loss += loss;
                    epoch_steps += 1;
                }
                Ok(None) => {} // batch had no NNUE features
                Err(e) => eprintln!("nnue_train_step error: {e}"),
            }
        }

        if epoch % 20 == 0 || epoch == 1 || epoch == epochs {
            let mean_loss =
                if epoch_steps > 0 { epoch_loss / epoch_steps as f32 } else { 0.0 };
            println!(
                "epoch {:4}/{epochs}  loss {mean_loss:.5}  ({:.1}s elapsed)",
                epoch,
                t0.elapsed().as_secs_f64()
            );
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
