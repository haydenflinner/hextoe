use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hextoe::game::GameState;
use hextoe::mcts::{Mcts, RandomRollout};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a game state with `n` moves already played at well-separated positions
/// so that no win is triggered and the candidate set is large.
fn make_state(n: usize) -> GameState {
    let mut state = GameState::new();
    // Place pieces in a spiral-ish pattern far enough apart to avoid wins.
    let positions: Vec<(i32, i32)> = (0..100)
        .flat_map(|i: i32| vec![(i * 3, 0), (0, i * 3), (i * 3, -i * 3)])
        .collect();
    for &pos in positions.iter().take(n) {
        state.place(pos);
    }
    state
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_place(c: &mut Criterion) {
    c.bench_function("place_single_on_empty", |b| {
        b.iter(|| {
            let mut s = GameState::new();
            s.place((0, 0))
        })
    });

    let state_20 = make_state(20);
    c.bench_function("place_single_20_pieces", |b| {
        b.iter(|| {
            let mut s = state_20.clone();
            s.place((1, 1))
        })
    });
}

fn bench_legal_actions(c: &mut Criterion) {
    let mut group = c.benchmark_group("legal_actions");
    for pieces in [0usize, 10, 20, 40].iter() {
        let state = make_state(*pieces);
        group.bench_with_input(
            BenchmarkId::from_parameter(pieces),
            &state,
            |b, s| b.iter(|| s.legal_actions()),
        );
    }
    group.finish();
}

fn bench_mcts(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_search_iters");
    for &iters in &[100u32, 500, 1_000] {
        group.bench_with_input(BenchmarkId::from_parameter(iters), &iters, |b, &n| {
            b.iter(|| {
                let mut rng = rand::thread_rng();
                let mut mcts = Mcts::new(GameState::new());
                let rollout = RandomRollout;
                mcts.search_iters(n, &mut rng, &rollout);
            })
        });
    }
    group.finish();
}

fn bench_mcts_best_moves(c: &mut Criterion) {
    // Pre-warm a tree with 1000 iterations; then benchmark best_moves extraction.
    let mut rng = rand::thread_rng();
    let mut mcts = Mcts::new(GameState::new());
    let rollout = RandomRollout;
    mcts.search_iters(1_000, &mut rng, &rollout);
    c.bench_function("best_moves_top3", |b| b.iter(|| mcts.best_moves(3)));
}

criterion_group!(
    benches,
    bench_place,
    bench_legal_actions,
    bench_mcts,
    bench_mcts_best_moves,
);
criterion_main!(benches);
