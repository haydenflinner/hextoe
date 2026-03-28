use ndarray::{Array1, Array2, Array4};
use numpy::{PyArray1, PyArray2, PyArray4, ToPyArray};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};

use hextoe::encode::{CHANNELS, GRID};
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

#[pymodule]
fn hextoe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sampler>()?;
    m.add("GRID", GRID)?;
    m.add("CHANNELS", CHANNELS)?;
    Ok(())
}
