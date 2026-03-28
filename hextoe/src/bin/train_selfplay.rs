//! Self-play training loop with champion promotion and optional human-game anchoring.
//!
//! Usage:
//!   hextoe-train-selfplay [flags] [game1.json game2.json ...]
//!
//! Any positional (non-flag) arguments are loaded as human game JSON files and used
//! as a permanent anchor dataset mixed into every training batch.  Shell glob expansion
//! works naturally:
//!   hextoe-train-selfplay --mcts 100 ~/Downloads/ih3t-games-chunk-*.json
//!
//! Maintains two checkpoints:
//!   --out PATH   (nnue_model.safetensors)  current training weights, saved every N seconds
//!   --best PATH  (nnue_best.safetensors)   champion, only updated when current beats it
//!
//! Each iteration:
//!   1. Play --games-per-iter games: --naive-frac vs naive, rest vs best champion.
//!   2. Train --train-steps gradient steps, mixing human records with self-play.
//!   3. Every --eval-every iters:
//!        a. Eval current vs naive
//!        b. Eval current vs best  → promote if >= --promote-rate for --promote-streak evals
//!   4. Save current weights every --save-interval seconds.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::Optimizer;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hextoe::encode::GRID;
use hextoe::game::Player;
use hextoe::nnue::{build_nnue_model, NNUENet, NNUERollout, NNUEWeights, DEFAULT_NNUE_PATH};
use hextoe::self_play::{GameRecord, SelfPlayCollector};
use hextoe::supervised::load_nnue_records_multi;

const DEFAULT_BEST_PATH: &str = "nnue_best.safetensors";

/// Label smoothing ε applied to human one-hot policy targets.
const LABEL_SMOOTH_EPS: f32 = 0.1;
/// Blend weight for heuristic priors when the position has threats (0 = ignore heuristics).
const HEURISTIC_BLEND_ALPHA: f32 = 0.3;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mcts_iters         = parse_arg(&args, "--mcts",             50u32);
    let games_per          = parse_arg(&args, "--games-per-iter",   10usize);
    let naive_frac         = parse_arg_f64(&args, "--naive-frac",   0.4);
    let train_steps        = parse_arg(&args, "--train-steps",      20usize);
    let batch_size         = parse_arg(&args, "--batch-size",       256usize);
    let lr                 = parse_arg_f64(&args, "--lr",           3e-4);
    let eval_games         = parse_arg(&args, "--eval-games",       40usize);
    let eval_every         = parse_arg(&args, "--eval-every",       5usize);
    let promote_rate       = parse_arg_f64(&args, "--promote-rate", 0.55);
    let promote_streak_req = parse_arg(&args, "--promote-streak",   2usize);
    let buf_cap            = parse_arg(&args, "--buffer",           20_000usize);
    let save_interval      = parse_arg(&args, "--save-interval",    300u64);
    let human_frac_cfg     = parse_arg_f64(&args, "--human-frac",   -1.0); // -1 = auto
    let policy_weight      = parse_arg_f64(&args, "--policy-weight", 3.0);
    let out_path           = parse_arg_str(&args, "--out",          DEFAULT_NNUE_PATH);
    let best_path          = parse_arg_str(&args, "--best",         DEFAULT_BEST_PATH);

    let naive_games = ((games_per as f64 * naive_frac).round() as usize).max(1);
    let self_games  = games_per.saturating_sub(naive_games);

    let device = Device::Cpu;

    // ── Load human game records (permanent anchor) ──────────────────────────
    let human_paths = collect_data_paths(&args);
    let human_records = if human_paths.is_empty() {
        println!("No human game files provided — training on self-play only.");
        vec![]
    } else {
        println!("Loading {} human game file(s)...", human_paths.len());
        match load_nnue_records_multi(&human_paths) {
            Ok((recs, used, skipped)) => {
                println!("Human data: {used} games used, {skipped} skipped → {} positions", recs.len());
                recs
            }
            Err(e) => {
                eprintln!("Failed to load human games: {e}");
                vec![]
            }
        }
    };

    // Fraction of each batch drawn from human records.
    let human_frac_batch: f64 = if human_frac_cfg >= 0.0 {
        human_frac_cfg
    } else if !human_records.is_empty() {
        0.5
    } else {
        0.0
    };
    let hu_per_batch = ((batch_size as f64 * human_frac_batch).round() as usize).min(batch_size);
    let sp_per_batch = batch_size - hu_per_batch;

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
    if Path::new(&best_path).exists() {
        best_varmap.load(&best_path).ok();
        println!("Loaded best weights from {best_path}");
    } else if Path::new(&out_path).exists() {
        best_varmap.load(&out_path).ok();
        println!("No best checkpoint — seeding best from current weights");
    } else {
        println!("No checkpoints found — starting from scratch");
    }
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
        "\nSelf-play training\n  mcts={mcts_iters}  {naive_games} naive + {self_games} self games/iter\n  \
         train_steps={train_steps}  batch={batch_size} ({hu_per_batch} human + {sp_per_batch} self-play)\n  \
         lr={lr:.1e}  eval every {eval_every} iters ({eval_games} games each)\n  \
         promote at {:.0}% for {promote_streak_req} evals  save every {save_interval}s\n",
        promote_rate * 100.0
    );

    for iter in 1.. {
        let t_iter = Instant::now();

        // ── Build rollouts ───────────────────────────────────────────────────
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
                Some(Player::X) if  cur_is_x => self_wins  += 1,
                Some(Player::O) if !cur_is_x => self_wins  += 1,
                Some(_)                      => self_losses += 1,
                None => {}
            }
            for rec in records {
                if buffer.len() >= buf_cap { buffer.pop_front(); }
                buffer.push_back(rec);
            }
        }

        // ── Training ─────────────────────────────────────────────────────────
        let sp_usable: Vec<&GameRecord> =
            buffer.iter().filter(|r| !r.nnue_feats.is_empty()).collect();

        let can_train = sp_usable.len() >= sp_per_batch.max(1)
            || (sp_per_batch == 0 && human_records.len() >= batch_size);

        let mut total_loss = 0.0f32;
        let mut total_policy_loss = 0.0f32;
        let mut loss_steps = 0usize;

        if can_train {
            let mut sp_idx: Vec<usize> = (0..sp_usable.len()).collect();
            let mut hu_idx: Vec<usize> = (0..human_records.len()).collect();

            for _ in 0..train_steps {
                sp_idx.shuffle(&mut rng);
                hu_idx.shuffle(&mut rng);

                let mut feats: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
                let mut z: Vec<f32> = Vec::with_capacity(batch_size);

                // Self-play portion
                let sp_take = sp_per_batch.min(sp_usable.len());
                for &i in &sp_idx[..sp_take] {
                    feats.push(sp_usable[i].nnue_feats.iter().map(|&f| f as usize).collect());
                    z.push(sp_usable[i].outcome);
                }

                // Human portion — also collect (batch_row, move_idx, record_idx) for policy supervision.
                let hu_take = hu_per_batch.min(human_records.len());
                let mut hu_pol_pairs: Vec<(u32, usize, usize)> = Vec::new(); // (batch_row, move_idx, hu_rec_idx)
                for (j, &i) in hu_idx[..hu_take].iter().enumerate() {
                    feats.push(human_records[i].feats.iter().map(|&f| f as usize).collect());
                    z.push(human_records[i].outcome);
                    if let Some(midx) = human_records[i].move_idx {
                        hu_pol_pairs.push(((sp_take + j) as u32, midx as usize, i));
                    }
                }

                if feats.is_empty() { continue; }
                let b = feats.len();

                let result = (|| -> candle_core::Result<(f32, f32)> {
                    let input  = NNUENet::dense_from_sparse(&feats, &device)?;
                    let target = Tensor::from_slice(&z, (b, 1usize), &device)?;
                    let (value_output, policy_logits) = model.forward_value_and_policy(&input)?;
                    let value_loss = (&value_output - &target)?.sqr()?.mean_all()?;

                    // Policy loss: self-play soft targets + human one-hot targets.
                    let mut pol_row_ids: Vec<u32> = Vec::new();
                    let mut pol_targets_flat: Vec<f32> = Vec::new();

                    // SP rows: MCTS pi distributions
                    for (i, &si) in sp_idx[..sp_take].iter().enumerate() {
                        pol_row_ids.push(i as u32);
                        pol_targets_flat.extend_from_slice(&*sp_usable[si].pi);
                    }

                    // Human rows: label-smoothed one-hot, blended with heuristic priors if available.
                    for &(row, midx, hi) in &hu_pol_pairs {
                        pol_row_ids.push(row);

                        // Label-smoothed one-hot: (1-ε) on human move, ε/(N-1) elsewhere.
                        let uniform_eps = LABEL_SMOOTH_EPS / (GRID * GRID - 1) as f32;
                        let mut target = vec![uniform_eps; GRID * GRID];
                        target[midx] = 1.0 - LABEL_SMOOTH_EPS;

                        // Heuristic blend: when the position has threats, pull the target
                        // toward the threat-aware distribution so the model isn't penalised
                        // for preferring critical blocks over the human's choice.
                        if !human_records[hi].heuristic_pi.is_empty() {
                            for v in target.iter_mut() { *v *= 1.0 - HEURISTIC_BLEND_ALPHA; }
                            for &(hidx, prob) in &human_records[hi].heuristic_pi {
                                target[hidx as usize] += HEURISTIC_BLEND_ALPHA * prob;
                            }
                        }

                        pol_targets_flat.extend_from_slice(&target);
                    }

                    let policy_loss_val = if pol_row_ids.is_empty() {
                        Tensor::zeros((), candle_core::DType::F32, &device)?
                    } else {
                        let n = pol_row_ids.len();
                        let idx_t = Tensor::from_vec(pol_row_ids, (n,), &device)?;
                        let sel_logits = policy_logits.index_select(&idx_t, 0)?;
                        let log_probs = log_softmax(&sel_logits, 1)?;
                        let pi_t = Tensor::from_vec(pol_targets_flat, (n, GRID * GRID), &device)?;
                        let ce = (pi_t * log_probs)?.sum(1)?.mean_all()?;
                        (ce * (-1.0f64))?
                    };

                    let scaled_policy_loss = (&policy_loss_val * policy_weight)?;
                    let loss = (&value_loss + &scaled_policy_loss)?;
                    opt.backward_step(&loss)?;
                    Ok((value_loss.to_scalar::<f32>()?, policy_loss_val.to_scalar::<f32>()?))
                })();
                match result {
                    Ok((vl, pl)) => { total_loss += vl; total_policy_loss += pl; loss_steps += 1; }
                    Err(e) => eprintln!("train step error: {e}"),
                }
            }
        }

        let mean_loss = if loss_steps > 0 { total_loss / loss_steps as f32 } else { f32::NAN };
        let mean_policy_loss = if loss_steps > 0 { total_policy_loss / loss_steps as f32 } else { f32::NAN };
        let iter_secs = t_iter.elapsed().as_secs_f64();

        println!(
            "iter {:4}  buf {:5}  naive W:{naive_wins} L:{naive_losses}  self W:{self_wins} L:{self_losses}  val_loss {:.4}  pol_loss {:.4}  {:.1}s",
            iter, buffer.len(), mean_loss, mean_policy_loss, iter_secs
        );

        // ── Time-based save ──────────────────────────────────────────────────
        if last_save.elapsed().as_secs() >= save_interval {
            if let Err(e) = varmap.save(&out_path) {
                eprintln!("save failed: {e}");
            } else {
                println!("  Saved current → {out_path}");
            }
            last_save = Instant::now();
        }

        // ── Eval ─────────────────────────────────────────────────────────────
        if iter % eval_every == 0 {
            let eval_cur_w = Arc::new(NNUEWeights::from_varmap(&varmap).expect("weights"));
            let eval_cur   = NNUERollout::new(eval_cur_w);
            let eval_best  = NNUERollout::new(best_weights.clone());

            // vs naive
            let (mut vn_w, mut vn_l, mut vn_d) = (0, 0, 0);
            for g in 0..eval_games {
                let naive_player = if g % 2 == 0 { Player::X } else { Player::O };
                let nnue_player  = naive_player.other();
                match collector.eval_game_vs_naive(mcts_iters, &mut rng, &eval_cur, naive_player) {
                    Some(w) if w == nnue_player => vn_w += 1,
                    Some(_) => vn_l += 1,
                    None    => vn_d += 1,
                }
            }
            let naive_rate = vn_w as f64 / eval_games as f64;

            // vs best
            let (mut vb_w, mut vb_l, mut vb_d) = (0, 0, 0);
            for g in 0..eval_games {
                let cur_is_x = g % 2 == 0;
                let winner = if cur_is_x {
                    collector.eval_game_two_rollouts(mcts_iters, &mut rng, &eval_cur, &eval_best)
                } else {
                    collector.eval_game_two_rollouts(mcts_iters, &mut rng, &eval_best, &eval_cur)
                };
                match winner {
                    Some(Player::X) if  cur_is_x => vb_w += 1,
                    Some(Player::O) if !cur_is_x => vb_w += 1,
                    Some(_) => vb_l += 1,
                    None    => vb_d += 1,
                }
            }
            // Promotion uses decisive-game win rate (W/(W+L)) to avoid draw dilution.
            let vb_decisive = (vb_w + vb_l) as f64;
            let vs_best_rate = if vb_decisive > 0.0 { vb_w as f64 / vb_decisive } else { 0.5 };

            println!(
                "  ── EVAL  vs-naive {:.1}% (W:{vn_w}/L:{vn_l}/D:{vn_d})  \
                 vs-best {:.1}% decisive (W:{vb_w}/L:{vb_l}/D:{vb_d})  \
                 streak {promote_streak}/{promote_streak_req}  ({:.0}s total)",
                naive_rate * 100.0, vs_best_rate * 100.0,
                t_total.elapsed().as_secs_f64()
            );

            // Promote?
            if vs_best_rate >= promote_rate {
                promote_streak += 1;
                if promote_streak >= promote_streak_req {
                    match varmap.save(&best_path) {
                        Ok(()) => {
                            best_weights = Arc::new(
                                NNUEWeights::from_varmap(&varmap).expect("extract weights")
                            );
                            println!("  *** Promoted current → best! ({best_path}) ***");
                        }
                        Err(e) => eprintln!("  promote save failed: {e}"),
                    }
                    promote_streak = 0;
                } else {
                    println!("  Promotion streak: {promote_streak}/{promote_streak_req}");
                }
            } else {
                if promote_streak > 0 {
                    println!("  Promotion streak reset (was {promote_streak})");
                }
                promote_streak = 0;
            }

            if let Err(e) = varmap.save(&out_path) {
                eprintln!("save current failed: {e}");
            }
            last_save = Instant::now();
        }
    }
}

/// Collect positional arguments (not `--flags` and not their values).
fn collect_data_paths(args: &[String]) -> Vec<String> {
    let flags_with_values = [
        "--mcts", "--games-per-iter", "--naive-frac", "--train-steps", "--batch-size",
        "--lr", "--eval-games", "--eval-every", "--promote-rate", "--promote-streak",
        "--buffer", "--save-interval", "--human-frac", "--policy-weight", "--out", "--best",
    ];
    let mut paths = Vec::new();
    let mut i = 1usize; // skip argv[0]
    while i < args.len() {
        if flags_with_values.contains(&args[i].as_str()) {
            i += 2; // skip flag + value
        } else if args[i].starts_with("--") {
            i += 1; // unknown boolean flag
        } else {
            paths.push(args[i].clone());
            i += 1;
        }
    }
    paths
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}
fn parse_arg_f64(args: &[String], flag: &str, default: f64) -> f64 { parse_arg(args, flag, default) }
fn parse_arg_str(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or_else(|| default.to_string())
}
