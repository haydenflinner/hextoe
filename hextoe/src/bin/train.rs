//! hextoe-train — self-play reinforcement learning loop (AlphaZero style).
//!
//! Usage:  cargo run --release --bin hextoe-train [OPTIONS]
//!
//! For a live dashboard:  cargo run --release --bin hextoe-train-gui [OPTIONS]
//!
//! Hyperparameters: edit `DEFAULT_*` in `hextoe::train` (`src/train.rs`).

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

    if let Err(e) = run_training(
        TrainingConfig::default_with_cli_rollout(),
        None,
        true,
        None,
        cli_one_checkpoint(),
    ) {
        eprintln!("Fatal error: {e}");
        std::process::exit(1);
    }
}
