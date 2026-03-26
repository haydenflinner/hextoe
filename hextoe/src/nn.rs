/// AlphaZero-style combined policy+value network for Hextoe.
///
/// Architecture:
///   trunk  : Conv(4→64, 3×3) + ReLU, then 4 residual blocks (64 ch)
///   policy : Conv(64→2, 1×1) + ReLU → flatten → Linear(2·G²→G²)  (logits)
///   value  : Conv(64→1, 1×1) + ReLU → flatten → Linear(G²→256) + ReLU
///            → Linear(256→1) + Tanh
///
/// where G = encode::GRID = 21.
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use rand::Rng;

use crate::encode::{action_to_index, board_center, encode_state, index_to_action, CHANNELS, GRID};
use crate::game::{GameState, Player};
use crate::mcts::{RandomRollout, RolloutPolicy, MAX_GAME_MOVES};

const HIDDEN: usize = 64;
const RES_BLOCKS: usize = 4;
const POLICY_CH: usize = 2;
const VALUE_CH: usize = 1;
const VALUE_FC: usize = 256;

// ── Residual block ────────────────────────────────────────────────────────────

struct ResBlock {
    c1: Conv2d,
    c2: Conv2d,
}

impl ResBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(ResBlock {
            c1: conv2d(HIDDEN, HIDDEN, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(HIDDEN, HIDDEN, 3, cfg, vb.pp("c2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.c1.forward(x)?.relu()?;
        let out = self.c2.forward(&out)?;
        (out + x)?.relu()
    }
}

// ── Network ───────────────────────────────────────────────────────────────────

pub struct HextoeNet {
    init_conv: Conv2d,
    res: Vec<ResBlock>,
    // policy head
    p_conv: Conv2d,
    p_fc: Linear,
    // value head
    v_conv: Conv2d,
    v_fc1: Linear,
    v_fc2: Linear,
}

impl HextoeNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let pad1 = Conv2dConfig { padding: 1, ..Default::default() };
        let no_pad = Conv2dConfig::default();
        Ok(HextoeNet {
            init_conv: conv2d(CHANNELS, HIDDEN, 3, pad1, vb.pp("init"))?,
            res: (0..RES_BLOCKS)
                .map(|i| ResBlock::new(vb.pp(&format!("r{i}"))))
                .collect::<Result<_>>()?,
            p_conv: conv2d(HIDDEN, POLICY_CH, 1, no_pad, vb.pp("pc"))?,
            p_fc: linear(POLICY_CH * GRID * GRID, GRID * GRID, vb.pp("pf"))?,
            v_conv: conv2d(HIDDEN, VALUE_CH, 1, no_pad, vb.pp("vc"))?,
            v_fc1: linear(VALUE_CH * GRID * GRID, VALUE_FC, vb.pp("vf1"))?,
            v_fc2: linear(VALUE_FC, 1, vb.pp("vf2"))?,
        })
    }

    /// Forward pass.
    ///
    /// `x`: `[B, CHANNELS, GRID, GRID]`
    ///
    /// Returns `(policy_logits [B, GRID²], value [B, 1])`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Trunk
        let mut h = self.init_conv.forward(x)?.relu()?;
        for b in &self.res {
            h = b.forward(&h)?;
        }

        // Policy head
        let p = self.p_conv.forward(&h)?.relu()?;
        let (pb, pc, ph, pw) = p.dims4()?;
        let p = p.reshape((pb, pc * ph * pw))?;
        let policy = self.p_fc.forward(&p)?;

        // Value head
        let v = self.v_conv.forward(&h)?.relu()?;
        let (vb_, vc, vh, vw) = v.dims4()?;
        let v = v.reshape((vb_, vc * vh * vw))?;
        let v = self.v_fc1.forward(&v)?.relu()?;
        let value = self.v_fc2.forward(&v)?.tanh()?;

        Ok((policy, value))
    }

    /// Evaluate a single game position.
    ///
    /// Returns `(prior_probs, value)` where:
    /// - `prior_probs[i]` is the network's policy probability for grid index `i`
    ///   (masked and renormalised to zero outside legal moves)
    /// - `value` ∈ `[-1, 1]` from the *current player's* perspective
    pub fn evaluate_state(
        &self,
        state: &GameState,
        device: &Device,
    ) -> Result<(Vec<f32>, f32)> {
        let enc = encode_state(state);
        let t = Tensor::from_slice(&enc, (1usize, CHANNELS, GRID, GRID), device)?;
        let (logits, value_t) = self.forward(&t)?;

        // Mask logits: set non-legal positions to -inf, then softmax.
        let center = board_center(state);
        let legal = state.legal_actions();
        let mut mask = vec![f32::NEG_INFINITY; GRID * GRID];
        for pos in &legal {
            if let Some(idx) = action_to_index(*pos, center) {
                mask[idx] = 0.0;
            }
        }
        let mask_t = Tensor::from_slice(&mask, (1usize, GRID * GRID), device)?;
        let masked_logits = (logits + mask_t)?;
        let probs = candle_nn::ops::softmax(&masked_logits, 1)?;
        let probs_vec = probs.reshape((GRID * GRID,))?.to_vec1::<f32>()?;

        let v: f32 = value_t.reshape((1usize,))?.to_vec1::<f32>()?[0];

        Ok((probs_vec, v))
    }
}

/// Weights + network; keep together for inference (layers reference [`VarMap`] tensors).
pub struct LoadedNet {
    varmap: VarMap,
    pub net: HextoeNet,
}

impl LoadedNet {
    /// Build structure and load weights from `path` (e.g. `hextoe_model.safetensors`).
    pub fn try_load(path: &str, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let net = HextoeNet::new(vb)?;
        let mut s = LoadedNet { varmap, net };
        load_weights(&mut s.varmap, path)?;
        Ok(s)
    }
}

/// Run one MCTS rollout using the network policy until terminal (used by [`NeuralRollout`]).
pub fn neural_rollout_policy(
    net: &HextoeNet,
    device: &Device,
    mut state: GameState,
    root_player: Player,
    rng: &mut impl Rng,
) -> f32 {
    let mut ply = 0u32;
    while !state.is_terminal() {
        if ply >= MAX_GAME_MOVES {
            return 0.0;
        }
        let actions = state.legal_actions();
        if actions.is_empty() {
            break;
        }
        let center = board_center(&state);
        let pos = match net.evaluate_state(&state, device) {
            Ok((probs, _)) => {
                let idx = sample_policy_index(&probs, rng);
                let pos = index_to_action(idx, center);
                if actions.contains(&pos) {
                    pos
                } else {
                    actions[rng.gen_range(0..actions.len())]
                }
            }
            Err(_) => {
                let mut fallback = RandomRollout;
                return fallback.rollout(state, root_player, rng);
            }
        };
        state.place(pos);
        ply += 1;
    }
    match state.winner {
        Some(p) if p == root_player => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// Sample a flat action index from a masked policy vector (illegal entries are ~0).
pub fn sample_policy_index(probs: &[f32], rng: &mut impl Rng) -> usize {
    let sum: f32 = probs.iter().copied().filter(|p| p.is_finite() && *p > 0.0).sum();
    if !(sum > 0.0) {
        return rng.gen_range(0..probs.len());
    }
    let t = rng.gen::<f32>() * sum;
    let mut c = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        c += p.max(0.0);
        if c >= t {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// MCTS rollout policy: play out with the network policy until terminal (AlphaZero-style).
pub struct NeuralRollout<'a> {
    pub net: &'a HextoeNet,
    pub device: &'a Device,
}

impl RolloutPolicy for NeuralRollout<'_> {
    fn rollout(&mut self, state: GameState, root_player: Player, rng: &mut impl Rng) -> f32 {
        neural_rollout_policy(self.net, self.device, state, root_player, rng)
    }

    const PARALLEL_SAFE: bool = false;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Initialise a fresh model with random weights on `device`.
pub fn build_model(device: &Device) -> Result<(VarMap, HextoeNet)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = HextoeNet::new(vb)?;
    Ok((varmap, model))
}

/// Save weights to a safetensors file.
pub fn save_weights(varmap: &VarMap, path: &str) -> Result<()> {
    varmap.save(path)
}

/// Load weights from a safetensors file into an existing VarMap.
pub fn load_weights(varmap: &mut VarMap, path: &str) -> Result<()> {
    varmap.load(path)
}
