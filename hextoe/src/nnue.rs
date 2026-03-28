//! NNUE value estimator — pure-CPU inference for game-level parallelism in self-play.
//!
//! Architecture (value-only, no policy head):
//!   sparse binary input (1_264 features) → L1 (512, ClippedReLU)
//!     → L2 (64, ClippedReLU) → L3 (32, ClippedReLU) → tanh scalar output
//!
//! The accumulator for L1 is `bias + Σ active_feature_columns`, so adding or
//! removing a piece from the board only updates one column — O(L1_SIZE) work
//! instead of O(N_FEATURES × L1_SIZE).
//!
//! Feature encoding (always from X's perspective):
//!   Two binary features per cell within NNUE_RADIUS of the board centroid:
//!     feature_idx = cell_idx          (X occupies this cell)
//!     feature_idx = cell_idx + N_CELLS (O occupies this cell)
//!   Feature N_CELLS*2 + 0 = 1 if X is to move
//!   Feature N_CELLS*2 + 1 = 1 if this is the 2nd move of the current player's pair
//!   "relative" = offset from board centroid, within hex-radius NNUE_RADIUS.
//!
//! Output is value from X's perspective; negate for O as root_player.
//!
//! Training: use `build_nnue_model` + `NNUENet::dense_from_sparse` + the same
//! `train_step`-style loop. The Candle model trains normally on CPU/Metal.
//! After training, call `NNUEWeights::load(path, device)` to get a fast
//! inference-only struct that runs entirely on plain Vec<f32>.

use std::collections::HashMap;
use std::sync::OnceLock;

use candle_core::{Device, DType, Result as CResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use rand::Rng;

use crate::encode::board_center;
use crate::game::{max_run_through, GameState, Player, Pos};
use crate::mcts::{move_weight, RolloutPolicy};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const NNUE_RADIUS: i32 = 14;
/// Hex cells within radius 14: 1 + 3·14·15 = 631.
pub const N_CELLS: usize = 1 + 3 * (NNUE_RADIUS as usize) * (NNUE_RADIUS as usize + 1);
/// Two turn-indicator features: "X to move" and "2nd of pair".
pub const N_TURN_FEATURES: usize = 2;
/// Total input features: one bit per (cell, player) + 2 turn bits.
pub const N_FEATURES: usize = N_CELLS * 2 + N_TURN_FEATURES;
pub const L1_SIZE: usize = 512;
pub const L2_SIZE: usize = 64;
pub const L3_SIZE: usize = 32;

pub const DEFAULT_NNUE_PATH: &str = "nnue_model.safetensors";

// ── Cell index table ──────────────────────────────────────────────────────────

fn build_cell_table() -> HashMap<Pos, usize> {
    let r = NNUE_RADIUS;
    let mut cells: Vec<Pos> = Vec::with_capacity(N_CELLS);
    for dq in -r..=r {
        let dr_lo = (-r).max(-dq - r);
        let dr_hi = r.min(-dq + r);
        for dr in dr_lo..=dr_hi {
            cells.push((dq, dr));
        }
    }
    cells.sort_by_key(|&(q, r)| (r, q));
    assert_eq!(cells.len(), N_CELLS, "N_CELLS formula mismatch");
    cells.into_iter().enumerate().map(|(i, p)| (p, i)).collect()
}

fn cell_table() -> &'static HashMap<Pos, usize> {
    static TABLE: OnceLock<HashMap<Pos, usize>> = OnceLock::new();
    TABLE.get_or_init(build_cell_table)
}

// ── Feature encoding ──────────────────────────────────────────────────────────

/// Return the active feature indices for `state`, relative to `center`.
///
/// One binary feature per (cell, player): index `cell_idx` for X, `cell_idx + N_CELLS` for O.
/// Pieces outside NNUE_RADIUS of `center` are silently dropped.
pub fn encode_nnue(state: &GameState, center: Pos) -> Vec<usize> {
    let table = cell_table();
    let mut features = Vec::with_capacity(state.board.len() + N_TURN_FEATURES);
    for (&(q, r), &player) in &state.board {
        let rel = (q - center.0, r - center.1);
        if let Some(&cell_idx) = table.get(&rel) {
            let offset = if player == Player::X { 0 } else { N_CELLS };
            features.push(offset + cell_idx);
        }
    }
    // Turn indicator features.
    if state.current_player() == Player::X {
        features.push(N_CELLS * 2);       // "X to move"
    }
    if state.total_moves > 0 && (state.total_moves - 1) % 2 == 1 {
        features.push(N_CELLS * 2 + 1);   // "2nd move of current pair"
    }
    features.sort_unstable();
    features
}

// ── Candle model (training) ───────────────────────────────────────────────────

pub struct NNUENet {
    fc0: Linear,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl NNUENet {
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            fc0: linear(N_FEATURES, L1_SIZE, vb.pp("fc0"))?,
            fc1: linear(L1_SIZE, L2_SIZE, vb.pp("fc1"))?,
            fc2: linear(L2_SIZE, L3_SIZE, vb.pp("fc2"))?,
            fc3: linear(L3_SIZE, 1, vb.pp("fc3"))?,
        })
    }

    /// Forward on dense input `[batch, N_FEATURES]`. Returns value `[batch, 1]` in `[-1,1]`.
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.fc0.forward(x)?.clamp(0f32, 1f32)?;
        let x = self.fc1.forward(&x)?.clamp(0f32, 1f32)?;
        let x = self.fc2.forward(&x)?.clamp(0f32, 1f32)?;
        self.fc3.forward(&x)?.tanh()
    }

    /// Sparse forward pass for training — avoids the O(batch × N_FEATURES) dense matmul.
    ///
    /// Pads all feature lists in the batch to the same length, does ONE batched
    /// index_select into fc0.weight, then masks out padding and sums per sample.
    /// Intermediate tensor is [batch × max_active × L1_SIZE] rather than
    /// [batch × N_FEATURES × L1_SIZE], giving a ~50× speedup when N_FEATURES is large.
    pub fn forward_sparse(&self, features_batch: &[Vec<usize>], device: &Device) -> CResult<Tensor> {
        let b = features_batch.len();
        let max_len = features_batch.iter().map(|f| f.len()).max().unwrap_or(0).max(1);

        // Build padded index [b × max_len] and float mask [b, max_len, 1].
        // Padding entries use index 0 and are zeroed out by the mask.
        let mut idx_data  = vec![0u32;  b * max_len];
        let mut mask_data = vec![0.0f32; b * max_len];
        for (i, feats) in features_batch.iter().enumerate() {
            for (j, &f) in feats.iter().enumerate() {
                idx_data [i * max_len + j] = f as u32;
                mask_data[i * max_len + j] = 1.0;
            }
        }

        let idx  = Tensor::from_vec(idx_data,  (b * max_len,), device)?;
        let mask = Tensor::from_vec(mask_data, (b, max_len, 1), device)?;

        // w0_t: [N_FEATURES, L1_SIZE] — must be contiguous for index_select.
        let w0_t = self.fc0.weight().t()?.contiguous()?;
        let b0   = self.fc0.bias().expect("fc0 bias");

        // ONE index_select: [b*max_len, L1_SIZE] → [b, max_len, L1_SIZE]
        let rows = w0_t.index_select(&idx, 0)?.reshape((b, max_len, L1_SIZE))?;
        // Masked sum: broadcast mask [b, max_len, 1] → [b, max_len, L1_SIZE], then sum.
        let acc  = rows.broadcast_mul(&mask)?.sum(1)?.broadcast_add(&b0.unsqueeze(0)?)?.clamp(0f32, 1f32)?;

        let l2 = self.fc1.forward(&acc)?.clamp(0f32, 1f32)?;
        let l3 = self.fc2.forward(&l2)?.clamp(0f32, 1f32)?;
        self.fc3.forward(&l3)?.tanh()
    }

    /// Build a dense `[batch, N_FEATURES]` tensor from a batch of sparse feature lists.
    /// Kept for compatibility; prefer `forward_sparse` for training.
    pub fn dense_from_sparse(features_batch: &[Vec<usize>], device: &Device) -> CResult<Tensor> {
        let batch = features_batch.len();
        let mut data = vec![0.0f32; batch * N_FEATURES];
        for (b, feats) in features_batch.iter().enumerate() {
            for &f in feats {
                data[b * N_FEATURES + f] = 1.0;
            }
        }
        Tensor::from_vec(data, (batch, N_FEATURES), device)
    }
}

pub fn build_nnue_model(device: &Device) -> CResult<(VarMap, NNUENet)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let net = NNUENet::new(vb)?;
    Ok((varmap, net))
}

// ── Fast CPU weights (inference) ──────────────────────────────────────────────

/// Weights extracted from a trained `NNUENet` for allocation-free CPU inference.
///
/// L0 is stored **column-major**: `w0[feat * L1_SIZE .. (feat+1)*L1_SIZE]` is the
/// column of output weights for that feature, so adding a piece = adding one slice.
pub struct NNUEWeights {
    w0: Vec<f32>, // column-major [N_FEATURES, L1_SIZE]
    b0: Vec<f32>, // [L1_SIZE]
    w1: Vec<f32>, // row-major   [L2_SIZE, L1_SIZE]
    b1: Vec<f32>, // [L2_SIZE]
    w2: Vec<f32>, // row-major   [L3_SIZE, L2_SIZE]
    b2: Vec<f32>, // [L3_SIZE]
    w3: Vec<f32>, // [L3_SIZE]  (output neuron weights)
    b3: f32,
}

impl NNUEWeights {
    fn build_from_named<F>(get: F) -> Result<Self, Box<dyn std::error::Error>>
    where
        F: Fn(&str) -> Result<Vec<f32>, Box<dyn std::error::Error>>,
    {
        let w0_row = get("fc0.weight")?;
        let mut w0_col = vec![0.0f32; N_FEATURES * L1_SIZE];
        for out_i in 0..L1_SIZE {
            for in_j in 0..N_FEATURES {
                w0_col[in_j * L1_SIZE + out_i] = w0_row[out_i * N_FEATURES + in_j];
            }
        }
        Ok(Self {
            w0: w0_col,
            b0: get("fc0.bias")?,
            w1: get("fc1.weight")?,
            b1: get("fc1.bias")?,
            w2: get("fc2.weight")?,
            b2: get("fc2.bias")?,
            w3: get("fc3.weight")?,
            b3: get("fc3.bias")?[0],
        })
    }

    /// Extract weights directly from a [`VarMap`] (no disk I/O).
    pub fn from_varmap(varmap: &VarMap) -> Result<Self, Box<dyn std::error::Error>> {
        let data = varmap.data().lock().unwrap();
        Self::build_from_named(|name| {
            let var = data
                .get(name)
                .ok_or_else(|| format!("NNUE: missing variable '{name}' in VarMap"))?;
            Ok(var.flatten_all()?.to_vec1::<f32>()?)
        })
    }

    /// Load a saved safetensors file and extract weights into column-major format.
    pub fn load(path: &str, device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let tensors = candle_core::safetensors::load(path, device)?;
        Self::build_from_named(|name| {
            let t = tensors
                .get(name)
                .ok_or_else(|| format!("NNUE weight '{name}' not found in {path}"))?;
            Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
        })
    }

    /// Evaluate a position from its sparse feature list.
    ///
    /// Returns value from X's perspective in `[-1, 1]`.
    pub fn eval_sparse(&self, features: &[usize]) -> f32 {
        // ── L0 accumulator ────────────────────────────────────────────────────
        let mut acc = self.b0.clone();
        for &f in features {
            let col = &self.w0[f * L1_SIZE..(f + 1) * L1_SIZE];
            for (a, &w) in acc.iter_mut().zip(col) {
                *a += w;
            }
        }
        for a in acc.iter_mut() {
            *a = a.clamp(0.0, 1.0);
        }

        // ── L1 → L2 ──────────────────────────────────────────────────────────
        let mut l2 = self.b1.clone();
        for (out_i, val) in l2.iter_mut().enumerate() {
            let row = &self.w1[out_i * L1_SIZE..(out_i + 1) * L1_SIZE];
            *val += acc.iter().zip(row).map(|(&a, &w)| a * w).sum::<f32>();
        }
        for v in l2.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }

        // ── L2 → L3 ──────────────────────────────────────────────────────────
        let mut l3 = self.b2.clone();
        for (out_i, val) in l3.iter_mut().enumerate() {
            let row = &self.w2[out_i * L2_SIZE..(out_i + 1) * L2_SIZE];
            *val += l2.iter().zip(row).map(|(&a, &w)| a * w).sum::<f32>();
        }
        for v in l3.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }

        // ── L3 → output ──────────────────────────────────────────────────────
        let mut out = self.b3;
        for (i, &v) in l3.iter().enumerate() {
            out += v * self.w3[i];
        }
        out.tanh()
    }
}

// ── RolloutPolicy implementation ──────────────────────────────────────────────

/// MCTS rollout policy backed by the NNUE value network.
///
/// Purely CPU-based, stateless and `Send + Sync`, so multiple games can run in
/// parallel (unlike the Candle/Metal `NeuralRollout`). No policy head → UCB1
/// selection (same as `RandomRollout` for tree policy), NNUE only provides the
/// leaf value.
pub struct NNUERollout {
    pub weights: std::sync::Arc<NNUEWeights>,
}

impl NNUERollout {
    pub fn new(weights: std::sync::Arc<NNUEWeights>) -> Self {
        Self { weights }
    }

    fn eval_state(&self, state: &GameState, root_player: Player) -> f32 {
        let center = board_center(state);
        let feats = encode_nnue(state, center);
        let value_x = self.weights.eval_sparse(&feats);
        if root_player == Player::X { value_x } else { -value_x }
    }
}

impl RolloutPolicy for NNUERollout {
    fn rollout(
        &self,
        state: GameState,
        root_player: Player,
        _rng: &mut impl Rng,
    ) -> (f32, Option<Vec<(Pos, f32)>>) {
        if state.is_terminal() {
            let v = match state.winner {
                Some(p) if p == root_player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            return (v, None);
        }

        let me = state.current_player();
        let opp = me.other();

        // If the player to move has an immediate win, they will take it.
        if state.candidates.iter().any(|&p| max_run_through(&state.board, p, me) >= 6) {
            let v = if me == root_player { 0.98 } else { -0.98 };
            return (v, None);
        }

        // If opponent has an immediate winning threat (we'll face it on their next turn),
        // the position is already very bad regardless of what we do.
        if state.candidates.iter().any(|&p| max_run_through(&state.board, p, opp) >= 6) {
            let v = if opp == root_player { 0.75 } else { -0.75 };
            return (v, None);
        }

        (self.eval_state(&state, root_player), None)
    }

    /// Return threat-weighted priors so PUCT focuses the search budget on the
    /// most tactically relevant moves first.  Uses the shared [`move_weight`]
    /// function which detects both linear runs and multi-axis threats (Triangle /
    /// Rhombus formations from Hex Tac-Toe theory).
    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        let actions = state.legal_actions();
        if actions.is_empty() {
            return None;
        }
        let me = state.current_player();
        let opp = me.other();

        let raw: Vec<(Pos, f32)> = actions
            .iter()
            .map(|&pos| (pos, move_weight(&state.board, pos, me, opp)))
            .collect();

        let total: f32 = raw.iter().map(|(_, w)| w).sum();
        Some(raw.into_iter().map(|(p, w)| (p, w / total)).collect())
    }

    /// Serial within a single game tree so PUCT priors guide the search correctly.
    /// Game-level parallelism (multiple concurrent self-play games) still works fine.
    const PARALLEL_SAFE: bool = false;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_count_is_correct() {
        assert_eq!(cell_table().len(), N_CELLS);
    }

    #[test]
    fn origin_maps_to_some_cell() {
        assert!(cell_table().contains_key(&(0, 0)));
    }

    #[test]
    fn edge_cell_within_radius() {
        // (NNUE_RADIUS, 0) is on the boundary and should be in the table.
        assert!(cell_table().contains_key(&(NNUE_RADIUS, 0)));
    }

    #[test]
    fn outside_radius_not_in_table() {
        assert!(!cell_table().contains_key(&(NNUE_RADIUS + 1, 0)));
    }

    #[test]
    fn encode_empty_board_has_only_x_to_move_feature() {
        // Empty board, X to move (total_moves=0): "X to move" active; "2nd of pair" not.
        let state = GameState::new();
        let feats = encode_nnue(&state, (0, 0));
        assert_eq!(feats.len(), 1);
        assert!(feats.contains(&N_AXIS_FEATURES)); // "X to move"
    }

    #[test]
    fn encode_isolated_piece_activates_three_level0_features() {
        // An isolated X piece has run=1 on all 3 axes → activates exactly 3 features
        // (one per axis at threshold level 0).  After placing at (0,0) it's O's turn
        // (1st of pair), so no turn features.
        let mut state = GameState::new();
        state.place((0, 0)); // X anchor; total_moves=1 → O 1st of pair
        let feats = encode_nnue(&state, (0, 0));
        // Only run-level features (< N_AXIS_FEATURES); no turn features.
        let run_feats: Vec<_> = feats.iter().filter(|&&f| f < N_AXIS_FEATURES).collect();
        assert_eq!(run_feats.len(), 3, "isolated piece: one level-0 feature per axis");
        assert!(feats.iter().all(|&f| f < N_FEATURES), "all indices in range");
    }

    #[test]
    fn encode_run_activates_more_levels() {
        // Two adjacent X pieces form a run-of-2 on one axis.
        // total_moves=4 after placing anchor(X) + 2 O moves + 1 X move.
        let mut state = GameState::new();
        state.place((0, 0)); // X, total_moves=1
        state.place((10, 0)); state.place((11, 0)); // O pair, total_moves=3
        state.place((1, 0)); // X, total_moves=4 → X 2nd of pair
        // (0,0) and (1,0) are adjacent X pieces → run=2 on axis (1,0).
        let feats = encode_nnue(&state, (0, 0));
        // Confirm all indices valid.
        assert!(feats.iter().all(|&f| f < N_FEATURES));
        // X is still to move (2nd of pair) → both turn features should be active.
        assert!(feats.contains(&N_AXIS_FEATURES),       "X to move");
        assert!(feats.contains(&(N_AXIS_FEATURES + 1)), "2nd of pair");
    }

    #[test]
    fn random_weights_eval_is_bounded() {
        // Build an untrained NNUE model on CPU, extract weights, check eval is in [-1,1].
        let device = Device::Cpu;
        let (varmap, _net) = build_nnue_model(&device).unwrap();
        // Save to a temp file and load back as NNUEWeights.
        let tmp = std::env::temp_dir().join("nnue_test.safetensors");
        varmap.save(&tmp).unwrap();
        let weights = NNUEWeights::load(tmp.to_str().unwrap(), &device).unwrap();

        let mut state = GameState::new();
        state.place((0, 0));
        state.place((1, 0));
        state.place((2, 0));
        let center = board_center(&state);
        let feats = encode_nnue(&state, center);
        let v = weights.eval_sparse(&feats);
        assert!((-1.0..=1.0).contains(&v), "value {v} out of range");
    }
}
