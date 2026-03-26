//! hextoe-train — self-play reinforcement learning loop (AlphaZero style).
//!
//! Usage:  cargo run --release --bin hextoe-train [--random-rollout]
//!
//! For a live dashboard:  cargo run --release --bin hextoe-train-gui [--random-rollout]
//!
//! Hyperparameters: edit `DEFAULT_*` in `hextoe::train` (`src/train.rs`).

use hextoe::train::{run_training, TrainingConfig};

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "hextoe-train\n\n\
             Options:\n\
               --random-rollout, -r   MCTS simulations use fast random playouts instead of the NN\n\
               -h, --help             Show this help\n"
        );
        return;
    }

    if let Err(e) = run_training(TrainingConfig::default_with_cli_rollout(), None, true, None) {
        eprintln!("Fatal error: {e}");
        std::process::exit(1);
    }
}
