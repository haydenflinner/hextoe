//! Training pipeline throughput (self-play + optimizer step). Run:
//!   cargo bench -p hextoe --bench train_bench
//!
//! Use these to catch accidental slowdowns in MCTS, self-play, or `train_step`.

use candle_core::Device;
use candle_nn::{optim::AdamW, optim::ParamsAdamW, Optimizer, VarBuilder, VarMap};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hextoe::encode::{CHANNELS, GRID};
use hextoe::mcts::RandomRollout;
use hextoe::nn::HextoeNet;
use hextoe::self_play::{GameRecord, ReplayBuffer, SelfPlayCollector};
use hextoe::device::default_inference_device;
use hextoe::train::{self_play_until_duration, train_step, TrainingConfig};
use rand::SeedableRng;
use rand::Rng;

const MCTS_ITERS_BENCH: u32 = 64;

fn dummy_game_records(batch: usize) -> Vec<GameRecord> {
    let mut rng = rand::thread_rng();
    (0..batch)
        .map(|_| {
            let mut state_enc = [0.0f32; CHANNELS * GRID * GRID];
            for x in &mut state_enc {
                *x = rng.gen::<f32>() * 0.01;
            }
            let mut pi = [0.0f32; GRID * GRID];
            let i = rng.gen_range(0..(GRID * GRID));
            pi[i] = 1.0;
            GameRecord {
                state_enc: Box::new(state_enc),
                pi: Box::new(pi),
                outcome: if rng.gen_bool(0.5) { 1.0 } else { -1.0 },
            }
        })
        .collect()
}

fn bench_self_play_one_game(c: &mut Criterion) {
    let collector = SelfPlayCollector::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xfeed_face);
    let rollout = RandomRollout;
    c.bench_function("self_play_one_game_random_mcts64", |b| {
        b.iter(|| {
            let recs = collector.play_game(MCTS_ITERS_BENCH, &mut rng, &rollout);
            black_box(recs.len())
        });
    });
}

fn bench_self_play_wall_clock_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_play_until_duration_random");
    for parallel in [1usize, 4] {
        group.bench_with_input(
            BenchmarkId::from_parameter(parallel),
            &parallel,
            |b, p| {
                let mut config = TrainingConfig::from_defaults(true);
                config.self_play_secs = 0.15;
                config.mcts_iters_per_move = MCTS_ITERS_BENCH;
                config.self_play_parallel_games = *p;
                let collector = SelfPlayCollector::new();
                let mut rng = rand::rngs::StdRng::seed_from_u64(1);
                let rollout = RandomRollout;
                b.iter(|| {
                    let mut buffer = ReplayBuffer::new(10_000);
                    let (n, _secs, games) = self_play_until_duration(
                        &collector,
                        &config,
                        &mut rng,
                        &rollout,
                        &None,
                        false,
                        &None,
                        &mut buffer,
                    );
                    black_box((n, games, buffer.len()))
                });
            },
        );
    }
    group.finish();
}

fn bench_train_step(c: &mut Criterion) {
    let device = default_inference_device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = HextoeNet::new(vb).expect("model");
    let adam = ParamsAdamW {
        lr: 3e-4,
        weight_decay: 1e-4,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adam).expect("optimizer");
    let batch = dummy_game_records(128);
    let refs: Vec<&GameRecord> = batch.iter().collect();

    c.bench_function("train_step_batch128_cpu", |b| {
        b.iter(|| {
            let loss = train_step(&model, &refs, &device, &mut opt).expect("step");
            black_box(loss)
        });
    });
}

criterion_group!(
    benches,
    bench_self_play_one_game,
    bench_self_play_wall_clock_parallel,
    bench_train_step,
);
criterion_main!(benches);
