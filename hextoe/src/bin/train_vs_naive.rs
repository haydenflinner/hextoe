//! Train the NNUE exclusively against the naive greedy bot until it never loses.
//!
//! Usage:
//!   hextoe-train-vs-naive [--mcts N] [--games-per-iter N] [--train-steps N]
//!                         [--batch-size N] [--lr LR] [--eval-games N]
//!                         [--win-rate F] [--out PATH]
//!
//! Each iteration:
//!   1. Play `--games-per-iter` games (NNUE bot vs NaiveRollout, sides alternated).
//!   2. Run `--train-steps` gradient steps on the collected positions.
//!   3. Every `--eval-every` iters, play `--eval-games` pure-eval games and report win rate.
//!   4. Stop when win rate >= `--win-rate` (default 0.95) for `--win-streak` consecutive evals.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use hextoe::game::Player;
use hextoe::nnue::{build_nnue_model, NNUENet, NNUERollout, NNUEWeights, DEFAULT_NNUE_PATH};
use hextoe::self_play::{GameRecord, SelfPlayCollector};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mcts_iters   = parse_arg(&args, "--mcts",           50u32);
    let games_per    = parse_arg(&args, "--games-per-iter", 10usize);
    let train_steps  = parse_arg(&args, "--train-steps",    20usize);
    let batch_size   = parse_arg(&args, "--batch-size",     256usize);
    let lr           = parse_arg_f64(&args, "--lr",         3e-4);
    let eval_games   = parse_arg(&args, "--eval-games",     40usize);
    let eval_every   = parse_arg(&args, "--eval-every",     5usize);
    let win_rate_thr = parse_arg_f64(&args, "--win-rate",   1.00);
    let win_streak   = parse_arg(&args, "--win-streak",     3usize);
    let buf_cap      = parse_arg(&args, "--buffer",         20_000usize);
    let out_path     = parse_arg_str(&args, "--out",        DEFAULT_NNUE_PATH);

    let device = Device::Cpu;
    let (mut varmap, model) = build_nnue_model(&device).expect("build NNUE");

    if Path::new(&out_path).exists() {
        match varmap.load(&out_path) {
            Ok(()) => println!("Loaded weights from {out_path}"),
            Err(e) => println!("Could not load {out_path}: {e} — starting fresh"),
        }
    } else {
        println!("No checkpoint — starting fresh");
    }

    let adam = ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), adam).expect("optimizer");

    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut buffer: VecDeque<GameRecord> = VecDeque::with_capacity(buf_cap);
    let mut streak = 0usize;
    let mut best_rate = 0.0f64;
    let t_total = Instant::now();

    println!(
        "Training NNUE vs NaiveRollout  mcts={mcts_iters}  games/iter={games_per}  \
         train_steps={train_steps}  batch={batch_size}  lr={lr:.1e}  \
         eval every {eval_every} iters ({eval_games} games)  target win rate {:.0}%",
        win_rate_thr * 100.0
    );

    for iter in 1.. {
        let t_iter = Instant::now();

        // ── Re-extract NNUE weights for rollout ──────────────────────────────
        let weights = match NNUEWeights::from_varmap(&varmap) {
            Ok(w) => Arc::new(w),
            Err(e) => { eprintln!("weight extract failed: {e}"); continue; }
        };
        let rollout = NNUERollout::new(weights);

        // ── Self-play: NNUE vs Naive ─────────────────────────────────────────
        let mut sp_wins = 0usize;
        let mut sp_losses = 0usize;
        let mut sp_games = 0usize;
        for g in 0..games_per {
            let naive_player = if g % 2 == 0 { Player::X } else { Player::O };
            let nnue_player = naive_player.other();
            let (records, winner) = collector.play_game_vs_naive(mcts_iters, &mut rng, &rollout, naive_player);
            match winner {
                Some(w) if w == nnue_player => sp_wins += 1,
                Some(_) => sp_losses += 1,
                None => {}
            }
            sp_games += 1;

            for rec in records {
                if buffer.len() >= buf_cap { buffer.pop_front(); }
                buffer.push_back(rec);
            }
        }

        // ── Training ─────────────────────────────────────────────────────────
        let usable: Vec<&GameRecord> = buffer.iter().filter(|r| !r.nnue_feats.is_empty()).collect();
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
                    let input = NNUENet::dense_from_sparse(&feats, &device)?;
                    let target = Tensor::from_slice(&z, (b, 1usize), &device)?;
                    let output = model.forward(&input)?;
                    let loss = (&output - &target)?.sqr()?.mean_all()?;
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

        println!(
            "iter {:4}  buf {:5}  W:{sp_wins} L:{sp_losses} D:{} ({:3.0}% win)  loss {:.4}  {:.1}s",
            iter, buffer.len(),
            sp_games - sp_wins - sp_losses,
            100.0 * sp_wins as f64 / sp_games.max(1) as f64,
            mean_loss, iter_secs
        );

        // ── Eval ─────────────────────────────────────────────────────────────
        if iter % eval_every == 0 {
            let weights2 = Arc::new(NNUEWeights::from_varmap(&varmap).expect("weights"));
            let eval_rollout = NNUERollout::new(weights2);
            let mut wins = 0usize;
            let mut losses = 0usize;
            let mut draws = 0usize;
            for g in 0..eval_games {
                let naive_player = if g % 2 == 0 { Player::X } else { Player::O };
                let nnue_player = naive_player.other();
                match collector.eval_game_vs_naive(mcts_iters, &mut rng, &eval_rollout, naive_player) {
                    Some(w) if w == nnue_player => wins += 1,
                    Some(_) => losses += 1,
                    None => draws += 1,
                }
            }
            let rate = wins as f64 / eval_games as f64;
            println!(
                "  ── EVAL  W:{wins} L:{losses} D:{draws} / {eval_games}  win rate {:.1}%  (total {:.0}s)",
                rate * 100.0,
                t_total.elapsed().as_secs_f64()
            );

            if rate > best_rate {
                best_rate = rate;
                if let Err(e) = varmap.save(&out_path) {
                    eprintln!("save failed: {e}");
                } else {
                    println!("  Saved → {out_path}  (new best {:.1}%)", best_rate * 100.0);
                }
            }

            if rate >= win_rate_thr {
                streak += 1;
                println!("  Win-rate streak: {streak}/{win_streak}");
                if streak >= win_streak {
                    println!("Target reached! NNUE dominates naive bot. Done in {:.0}s.", t_total.elapsed().as_secs_f64());
                    break;
                }
            } else {
                streak = 0;
            }
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
