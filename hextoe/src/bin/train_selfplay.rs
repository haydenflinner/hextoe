//! Self-play training loop with champion promotion.
//!
//! Maintains two checkpoints:
//!   --out PATH   (nnue_model.safetensors)  — current training weights, saved every N minutes
//!   --best PATH  (nnue_best.safetensors)   — champion, only promoted when current beats it
//!
//! Each iteration:
//!   1. Play --games-per-iter games: fraction vs naive (easy signal), rest vs best (hard signal).
//!   2. Train --train-steps gradient steps.
//!   3. Every --eval-every iters, run evals:
//!        a. current vs naive  → report win rate
//!        b. current vs best   → report win rate; promote if >= --promote-rate for --promote-streak consecutive evals
//!   4. Save current weights every --save-interval seconds regardless of eval.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hextoe::game::Player;
use hextoe::nnue::{build_nnue_model, NNUENet, NNUERollout, NNUEWeights, DEFAULT_NNUE_PATH};
use hextoe::self_play::{GameRecord, SelfPlayCollector};

const DEFAULT_BEST_PATH: &str = "nnue_best.safetensors";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mcts_iters    = parse_arg(&args, "--mcts",             50u32);
    let games_per     = parse_arg(&args, "--games-per-iter",   10usize);
    let naive_frac    = parse_arg_f64(&args, "--naive-frac",   0.4);
    let train_steps   = parse_arg(&args, "--train-steps",      20usize);
    let batch_size    = parse_arg(&args, "--batch-size",       256usize);
    let lr            = parse_arg_f64(&args, "--lr",           3e-4);
    let eval_games    = parse_arg(&args, "--eval-games",       40usize);
    let eval_every    = parse_arg(&args, "--eval-every",       5usize);
    let promote_rate  = parse_arg_f64(&args, "--promote-rate", 0.55);
    let promote_streak_req = parse_arg(&args, "--promote-streak", 2usize);
    let buf_cap       = parse_arg(&args, "--buffer",           20_000usize);
    let save_interval = parse_arg(&args, "--save-interval",    300u64);  // seconds
    let out_path      = parse_arg_str(&args, "--out",          DEFAULT_NNUE_PATH);
    let best_path     = parse_arg_str(&args, "--best",         DEFAULT_BEST_PATH);

    let naive_games = ((games_per as f64 * naive_frac).round() as usize).max(1);
    let self_games  = games_per.saturating_sub(naive_games);

    let device = Device::Cpu;

    // ── Load current weights ────────────────────────────────────────────────
    let (mut varmap, model) = build_nnue_model(&device).expect("build NNUE");
    if Path::new(&out_path).exists() {
        match varmap.load(&out_path) {
            Ok(()) => println!("Loaded current weights from {out_path}"),
            Err(e) => println!("Could not load {out_path}: {e} — starting fresh"),
        }
    } else {
        println!("No checkpoint at {out_path} — starting fresh");
    }

    // ── Load best weights ───────────────────────────────────────────────────
    let (mut best_varmap, _) = build_nnue_model(&device).expect("build best NNUE");
    let best_path_loaded = if Path::new(&best_path).exists() {
        best_varmap.load(&best_path).ok();
        println!("Loaded best weights from {best_path}");
        best_path.clone()
    } else if Path::new(&out_path).exists() {
        best_varmap.load(&out_path).ok();
        println!("No best checkpoint — initializing best from current weights");
        out_path.clone()
    } else {
        println!("No best checkpoint — starting best from scratch");
        String::new()
    };
    let _ = best_path_loaded;

    let mut best_weights = Arc::new(
        NNUEWeights::from_varmap(&best_varmap).expect("extract best weights")
    );

    let adam = ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), adam).expect("optimizer");

    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut buffer: VecDeque<GameRecord> = VecDeque::with_capacity(buf_cap);

    let mut promote_streak = 0usize;
    let mut last_save = Instant::now();
    let t_total = Instant::now();

    println!(
        "Self-play training  mcts={mcts_iters}  {naive_games} naive + {self_games} self games/iter  \
         train_steps={train_steps}  batch={batch_size}  lr={lr:.1e}\n\
         eval every {eval_every} iters ({eval_games} games each)  \
         promote at {:.0}% for {promote_streak_req} evals  save every {save_interval}s",
        promote_rate * 100.0
    );

    for iter in 1.. {
        let t_iter = Instant::now();

        // ── Build current rollout ────────────────────────────────────────────
        let cur_weights = match NNUEWeights::from_varmap(&varmap) {
            Ok(w) => Arc::new(w),
            Err(e) => { eprintln!("weight extract failed: {e}"); continue; }
        };
        let cur_rollout  = NNUERollout::new(cur_weights);
        let best_rollout = NNUERollout::new(best_weights.clone());

        // ── Self-play: current vs naive ──────────────────────────────────────
        let mut naive_wins = 0usize;
        let mut naive_losses = 0usize;
        for g in 0..naive_games {
            let naive_player = if g % 2 == 0 { Player::X } else { Player::O };
            let nnue_player  = naive_player.other();
            let (records, winner) =
                collector.play_game_vs_naive(mcts_iters, &mut rng, &cur_rollout, naive_player);
            match winner {
                Some(w) if w == nnue_player => naive_wins  += 1,
                Some(_)                     => naive_losses += 1,
                None => {}
            }
            for rec in records {
                if buffer.len() >= buf_cap { buffer.pop_front(); }
                buffer.push_back(rec);
            }
        }

        // ── Self-play: current vs best ───────────────────────────────────────
        let mut self_wins = 0usize;
        let mut self_losses = 0usize;
        for g in 0..self_games {
            let cur_is_x = g % 2 == 0;
            let (records, winner) = if cur_is_x {
                collector.play_game_two_rollouts(mcts_iters, &mut rng, &cur_rollout, &best_rollout)
            } else {
                collector.play_game_two_rollouts(mcts_iters, &mut rng, &best_rollout, &cur_rollout)
            };
            match winner {
                Some(Player::X) if  cur_is_x => self_wins   += 1,
                Some(Player::O) if !cur_is_x => self_wins   += 1,
                Some(_)                      => self_losses  += 1,
                None => {}
            }
            for rec in records {
                if buffer.len() >= buf_cap { buffer.pop_front(); }
                buffer.push_back(rec);
            }
        }

        // ── Training ─────────────────────────────────────────────────────────
        let usable: Vec<&GameRecord> =
            buffer.iter().filter(|r| !r.nnue_feats.is_empty()).collect();
        let mut total_loss = 0.0f32;
        let mut loss_steps = 0usize;
        if usable.len() >= batch_size {
            let mut idx: Vec<usize> = (0..usable.len()).collect();
            for _ in 0..train_steps {
                idx.shuffle(&mut rng);
                let batch: Vec<&GameRecord> = idx[..batch_size].iter().map(|&i| usable[i]).collect();
                let feats: Vec<Vec<usize>> = batch.iter()
                    .map(|r| r.nnue_feats.iter().map(|&f| f as usize).collect())
                    .collect();
                let z: Vec<f32> = batch.iter().map(|r| r.outcome).collect();
                let b = batch.len();
                let result = (|| -> candle_core::Result<f32> {
                    let input  = NNUENet::dense_from_sparse(&feats, &device)?;
                    let target = Tensor::from_slice(&z, (b, 1usize), &device)?;
                    let output = model.forward(&input)?;
                    let loss   = (&output - &target)?.sqr()?.mean_all()?;
                    opt.backward_step(&loss)?;
                    loss.to_scalar::<f32>()
                })();
                match result {
                    Ok(l) => { total_loss += l; loss_steps += 1; }
                    Err(e) => eprintln!("train step error: {e}"),
                }
            }
        }
        let mean_loss = if loss_steps > 0 { total_loss / loss_steps as f32 } else { f32::NAN };
        let iter_secs = t_iter.elapsed().as_secs_f64();

        let total_games = naive_games + self_games;
        println!(
            "iter {:4}  buf {:5}  naive W:{naive_wins} L:{naive_losses}  self W:{self_wins} L:{self_losses} / {total_games}  loss {:.4}  {:.1}s",
            iter, buffer.len(), mean_loss, iter_secs
        );

        // ── Time-based save of current ───────────────────────────────────────
        if last_save.elapsed().as_secs() >= save_interval {
            if let Err(e) = varmap.save(&out_path) {
                eprintln!("save current failed: {e}");
            } else {
                println!("  Saved current → {out_path}");
            }
            last_save = Instant::now();
        }

        // ── Eval ─────────────────────────────────────────────────────────────
        if iter % eval_every == 0 {
            let eval_cur_w = Arc::new(NNUEWeights::from_varmap(&varmap).expect("weights"));
            let eval_cur  = NNUERollout::new(eval_cur_w);
            let eval_best = NNUERollout::new(best_weights.clone());

            // vs naive
            let mut vn_wins = 0usize;
            let mut vn_losses = 0usize;
            let mut vn_draws = 0usize;
            for g in 0..eval_games {
                let naive_player = if g % 2 == 0 { Player::X } else { Player::O };
                let nnue_player  = naive_player.other();
                match collector.eval_game_vs_naive(mcts_iters, &mut rng, &eval_cur, naive_player) {
                    Some(w) if w == nnue_player => vn_wins   += 1,
                    Some(_)                     => vn_losses  += 1,
                    None                        => vn_draws   += 1,
                }
            }
            let naive_rate = vn_wins as f64 / eval_games as f64;

            // vs best
            let mut vb_wins = 0usize;
            let mut vb_losses = 0usize;
            let mut vb_draws = 0usize;
            for g in 0..eval_games {
                let cur_is_x = g % 2 == 0;
                let winner = if cur_is_x {
                    collector.eval_game_two_rollouts(mcts_iters, &mut rng, &eval_cur, &eval_best)
                } else {
                    collector.eval_game_two_rollouts(mcts_iters, &mut rng, &eval_best, &eval_cur)
                };
                match winner {
                    Some(Player::X) if  cur_is_x => vb_wins   += 1,
                    Some(Player::O) if !cur_is_x => vb_wins   += 1,
                    Some(_)                      => vb_losses  += 1,
                    None                         => vb_draws   += 1,
                }
            }
            let vs_best_rate = vb_wins as f64 / eval_games as f64;

            println!(
                "  ── EVAL  vs-naive: W:{vn_wins} L:{vn_losses} D:{vn_draws} ({:.1}%)  \
                 vs-best: W:{vb_wins} L:{vb_losses} D:{vb_draws} ({:.1}%)  \
                 promote streak {promote_streak}/{promote_streak_req}  (total {:.0}s)",
                naive_rate * 100.0,
                vs_best_rate * 100.0,
                t_total.elapsed().as_secs_f64()
            );

            // Promote?
            if vs_best_rate >= promote_rate {
                promote_streak += 1;
                println!("  Promotion streak: {promote_streak}/{promote_streak_req}");
                if promote_streak >= promote_streak_req {
                    match varmap.save(&best_path) {
                        Ok(()) => {
                            best_weights = Arc::new(
                                NNUEWeights::from_varmap(&varmap).expect("extract weights")
                            );
                            println!("  *** Promoted current → best! Saved to {best_path} ***");
                        }
                        Err(e) => eprintln!("  promote save failed: {e}"),
                    }
                    promote_streak = 0;
                }
            } else {
                if promote_streak > 0 {
                    println!("  Promotion streak reset (was {promote_streak})");
                }
                promote_streak = 0;
            }

            // Always save current at eval time too (overrides time-based save)
            if let Err(e) = varmap.save(&out_path) {
                eprintln!("save current failed: {e}");
            }
            last_save = Instant::now();
        }
    }
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn parse_arg_f64(args: &[String], flag: &str, default: f64) -> f64 { parse_arg(args, flag, default) }
fn parse_arg_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or_else(|| default.to_string())
}
