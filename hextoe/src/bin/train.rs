//! hextoe-train — self-play reinforcement learning loop (AlphaZero style).
//!
//! Usage:  cargo run --release --bin hextoe-train [OPTIONS]
//!
//! For a live dashboard:  cargo run --release --bin hextoe-train-gui [OPTIONS]
//!
//! Hyperparameters: edit `DEFAULT_*` in `hextoe::train` (`src/train.rs`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use hextoe::train::{cli_one_checkpoint, run_training, TrainingConfig};

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "hextoe-train\n\n\
             Options:\n\
               --random-rollout, -r   MCTS simulations use fast random playouts instead of NN leaf value\n\
               --population N, -p N   Population tournament mode with N candidates (default 1 = classic)\n\
               --one-checkpoint       Exit after writing latest checkpoint once (e.g. for cargo flamegraph)\n\
               -h, --help             Show this help\n"
        );
        return;
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();
    ctrlc::set_handler(move || {
        if cancel_clone.load(Ordering::Relaxed) {
            // Second CTRL+C: force-quit immediately.
            eprintln!("\nForce-quitting.");
            std::process::exit(1);
        }
        eprintln!("\nCTRL+C received — finishing current phase then saving. Press again to force-quit.");
        cancel_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting CTRL+C handler");

    if let Err(e) = run_training(
        TrainingConfig::default_with_cli_rollout(),
        None,
        true,
        Some(cancel),
        cli_one_checkpoint(),
    ) {
        eprintln!("Fatal error: {e}");
        std::process::exit(1);
    }
}
