use ndarray::{Array1, Array2, Array3, Array4};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, ToPyArray};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};

use hextoe::encode::{action_to_index, board_center, encode_state, index_to_action, CHANNELS, GRID};
use hextoe::game::{GameState, Player};
use hextoe::mcts::{compound_threat_priors, naive_best_move};
use hextoe::supervised::{build_sample_index, encode_sample, load_raw_games_multi, RawGame};

/// Data sampler backed by Rust game replay + encoding.
///
/// Stores only raw move sequences (~KBs for thousands of games).
/// Encodes board states on demand per batch, applying a random
/// D₆ symmetry transform each time for free augmentation.
#[pyclass]
struct Sampler {
    games: Vec<RawGame>,
    pairs: Vec<(usize, usize)>,
    rng: rand::rngs::StdRng,
}

#[pymethods]
impl Sampler {
    /// Load games from one or more JSON files.
    #[new]
    fn new(paths: Vec<String>) -> PyResult<Self> {
        let (games, used, skipped) =
            load_raw_games_multi(&paths).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
        let pairs = build_sample_index(&games);
        println!(
            "hextoe_py: {used} games loaded, {skipped} skipped → {} positions",
            pairs.len()
        );
        Ok(Sampler {
            games,
            pairs,
            rng: rand::rngs::StdRng::from_entropy(),
        })
    }

    /// Total number of (game, step) positions across all loaded games.
    fn __len__(&self) -> usize {
        self.pairs.len()
    }

    /// Sample `n` positions with replacement, each with a random symmetry transform.
    ///
    /// Returns a tuple of three numpy float32 arrays:
    ///   states   : shape [n, CHANNELS, GRID, GRID]
    ///   policies : shape [n, GRID*GRID]  — one-hot over played move
    ///   values   : shape [n]             — +1 winner, -1 loser
    fn sample_batch<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let g2 = GRID * GRID;
        let mut states_vec: Vec<f32> = Vec::with_capacity(n * CHANNELS * GRID * GRID);
        let mut policies_vec: Vec<f32> = Vec::with_capacity(n * g2);
        let mut values_vec: Vec<f32> = Vec::with_capacity(n);

        let mut sampled = 0usize;
        while sampled < n {
            let idx = self.rng.gen_range(0..self.pairs.len());
            let (gi, si) = self.pairs[idx];
            let tid = self.rng.gen_range(0u8..12);
            if let Some(rec) = encode_sample(&self.games, gi, si, tid) {
                states_vec.extend_from_slice(rec.state_enc.as_ref());
                policies_vec.extend_from_slice(rec.pi.as_ref());
                values_vec.push(rec.outcome);
                sampled += 1;
            }
        }

        let states = Array4::from_shape_vec((n, CHANNELS, GRID, GRID), states_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let policies = Array2::from_shape_vec((n, g2), policies_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let values = Array1::from_vec(values_vec);

        Ok((
            states.to_pyarray_bound(py),
            policies.to_pyarray_bound(py),
            values.to_pyarray_bound(py),
        ))
    }

    /// Like sample_batch but iterates through all positions in epoch order (shuffled).
    /// Returns a list of (states, policies, values) batches covering the full dataset once.
    fn epoch_batches<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<Vec<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )>> {
        use rand::seq::SliceRandom;
        let mut order: Vec<usize> = (0..self.pairs.len()).collect();
        order.shuffle(&mut self.rng);

        let g2 = GRID * GRID;
        let mut result = Vec::new();

        for chunk in order.chunks(batch_size) {
            if chunk.is_empty() {
                continue;
            }
            let mut states_vec: Vec<f32> = Vec::with_capacity(chunk.len() * CHANNELS * GRID * GRID);
            let mut policies_vec: Vec<f32> = Vec::with_capacity(chunk.len() * g2);
            let mut values_vec: Vec<f32> = Vec::with_capacity(chunk.len());
            let mut actual = 0usize;

            for &pi in chunk {
                let (gi, si) = self.pairs[pi];
                let tid = self.rng.gen_range(0u8..12);
                if let Some(rec) = encode_sample(&self.games, gi, si, tid) {
                    states_vec.extend_from_slice(rec.state_enc.as_ref());
                    policies_vec.extend_from_slice(rec.pi.as_ref());
                    values_vec.push(rec.outcome);
                    actual += 1;
                }
            }

            if actual == 0 {
                continue;
            }

            let states = Array4::from_shape_vec((actual, CHANNELS, GRID, GRID), states_vec)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let policies = Array2::from_shape_vec((actual, g2), policies_vec)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let values = Array1::from_vec(values_vec);

            result.push((
                states.to_pyarray_bound(py),
                policies.to_pyarray_bound(py),
                values.to_pyarray_bound(py),
            ));
        }

        Ok(result)
    }
}

/// Python-accessible wrapper around `GameState`.
#[pyclass]
struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    fn new() -> Self {
        PyGameState { inner: GameState::new() }
    }

    fn clone(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    /// Place a stone at axial (q, r). Returns True if legal.
    fn place(&mut self, q: i32, r: i32) -> bool {
        self.inner.place((q, r))
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Returns list of (q, r) tuples for all legal moves.
    fn legal_actions(&self) -> Vec<(i32, i32)> {
        self.inner.legal_actions()
    }

    /// 0 = Player X, 1 = Player O.
    fn current_player(&self) -> u8 {
        match self.inner.current_player() {
            Player::X => 0,
            Player::O => 1,
        }
    }

    /// Returns 0 (X won), 1 (O won), or -1 (no winner yet).
    fn winner(&self) -> i8 {
        match self.inner.winner {
            Some(Player::X) => 0,
            Some(Player::O) => 1,
            None => -1,
        }
    }

    /// Encode current state as numpy [CHANNELS, GRID, GRID] float32.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let flat = encode_state(&self.inner);
        let arr = Array3::from_shape_vec((CHANNELS, GRID, GRID), flat.to_vec()).unwrap();
        arr.to_pyarray_bound(py)
    }

    /// Board centroid (q, r) — used as origin for action index mapping.
    fn board_center(&self) -> (i32, i32) {
        board_center(&self.inner)
    }

    /// Convert axial position to flat grid index (row*GRID+col). Returns -1 if out of bounds.
    fn action_to_index(&self, q: i32, r: i32) -> i64 {
        let center = board_center(&self.inner);
        match action_to_index((q, r), center) {
            Some(idx) => idx as i64,
            None => -1,
        }
    }

    /// Convert flat grid index back to (q, r).
    fn index_to_action(&self, idx: usize) -> (i32, i32) {
        let center = board_center(&self.inner);
        index_to_action(idx, center)
    }

    /// Pick the best tactical (naive) move. Returns (q, r) or None if terminal.
    fn naive_move(&self) -> Option<(i32, i32)> {
        naive_best_move(&self.inner)
    }

    fn total_moves(&self) -> u32 {
        self.inner.total_moves
    }

    /// Returns normalized tactical prior weights for all legal actions.
    ///
    /// Uses compound-threat heuristics (blocking 4-in-a-row, 5-in-a-row, forks, etc.).
    /// Returns an empty list when there are no meaningful threats (all weights equal),
    /// so the caller can skip blending when the position is quiet.
    ///
    /// Each entry is ((q, r), weight) with weights summing to 1.0.
    fn tactical_priors(&self) -> Vec<((i32, i32), f32)> {
        let actions = self.inner.legal_actions();
        if actions.is_empty() {
            return vec![];
        }
        let me = self.inner.current_player();
        let opp = me.other();
        let raw = compound_threat_priors(&self.inner, &actions, me, opp);
        let max_w = raw.iter().map(|(_, w)| *w).fold(0.0f32, f32::max);
        if max_w <= 1.0 {
            return vec![];  // quiet position — no tactical override needed
        }
        let total: f32 = raw.iter().map(|(_, w)| w).sum();
        raw.into_iter().map(|(p, w)| (p, w / total)).collect()
    }
}

#[pymodule]
fn hextoe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sampler>()?;
    m.add_class::<PyGameState>()?;
    m.add("GRID", GRID)?;
    m.add("CHANNELS", CHANNELS)?;
    Ok(())
}
