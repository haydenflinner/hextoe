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
use crate::game::{GameState, Player, Pos};
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

    /// Shared trunk used by both heads.
    fn forward_trunk(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.init_conv.forward(x)?.relu()?;
        for b in &self.res {
            h = b.forward(&h)?;
        }
        Ok(h)
    }

    fn forward_policy_head_from_trunk(&self, h: &Tensor) -> Result<Tensor> {
        // Returns policy logits with shape [B, GRID * GRID].
        let p = self.p_conv.forward(h)?.relu()?;
        let (pb, pc, ph, pw) = p.dims4()?;
        let p = p.reshape((pb, pc * ph * pw))?;
        let policy = self.p_fc.forward(&p)?;
        Ok(policy)
    }

    fn forward_value_head_from_trunk(&self, h: &Tensor) -> Result<Tensor> {
        // Returns value with shape [B, 1] in [-1, 1].
        let v = self.v_conv.forward(h)?.relu()?;
        let (vb_, vc, vh, vw) = v.dims4()?;
        let v = v.reshape((vb_, vc * vh * vw))?;
        let v = self.v_fc1.forward(&v)?.relu()?;
        let value = self.v_fc2.forward(&v)?.tanh()?;
        Ok(value)
    }

    /// Forward pass.
    ///
    /// `x`: `[B, CHANNELS, GRID, GRID]`
    ///
    /// Returns `(policy_logits [B, GRID²], value [B, 1])`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.forward_trunk(x)?;
        let policy = self.forward_policy_head_from_trunk(&h)?;
        let value = self.forward_value_head_from_trunk(&h)?;
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
        let h = self.forward_trunk(&t)?;
        let logits = self.forward_policy_head_from_trunk(&h)?;
        let value_t = self.forward_value_head_from_trunk(&h)?;

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

    /// Evaluate a single game position, returning only the value head.
    ///
    /// This is used by MCTS leaf evaluation; we avoid computing the policy head and
    /// the legal-move mask/softmax to keep self-play/tournament fast.
    ///
    /// Returns `value` in `[-1, 1]` from the *current player's* perspective.
    pub fn evaluate_value_state(&self, state: &GameState, device: &Device) -> Result<f32> {
        let enc = encode_state(state);
        let t = Tensor::from_slice(&enc, (1usize, CHANNELS, GRID, GRID), device)?;
        let h = self.forward_trunk(&t)?;
        let value_t = self.forward_value_head_from_trunk(&h)?;
        let v: f32 = value_t.reshape((1usize,))?.to_vec1::<f32>()?[0];
        Ok(v)
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

/// Run one MCTS rollout using the network policy until terminal.
///
/// **Slow:** one full forward pass per ply until the game ends. Prefer
/// [`neural_leaf_value_policy`] for MCTS (AlphaZero-style: single value eval at the leaf).
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
            return state.board_heuristic(root_player);
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
                let fallback = RandomRollout;
                return fallback.rollout(state, root_player, rng).0;
            }
        };
        state.place(pos);
        ply += 1;
    }
    match state.winner {
        Some(p) if p == root_player => 1.0,
        Some(_) => -1.0,
        None => state.board_heuristic(root_player),
    }
}

/// One forward pass at the leaf: value from the network, converted to `root_player`'s
/// perspective (AlphaZero-style MCTS backup target).
pub fn neural_leaf_value_policy(
    net: &HextoeNet,
    device: &Device,
    state: GameState,
    root_player: Player,
    rng: &mut impl Rng,
) -> f32 {
    if state.is_terminal() {
        return match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }
    match net.evaluate_value_state(&state, device) {
        Ok(v) => {
            let cp = state.current_player();
            if cp == root_player { v } else { -v }
        }
        Err(_) => {
            let fallback = RandomRollout;
            fallback.rollout(state, root_player, rng).0
        }
    }
}

/// Extract PUCT priors for `state`'s legal actions from a network forward pass.
fn nn_priors(net: &HextoeNet, device: &Device, state: &GameState) -> Option<Vec<(Pos, f32)>> {
    let center = board_center(state);
    let (probs, _) = net.evaluate_state(state, device).ok()?;
    let legal = state.legal_actions();
    Some(
        legal
            .iter()
            .filter_map(|&pos| {
                let idx = action_to_index(pos, center)?;
                Some((pos, probs[idx]))
            })
            .collect(),
    )
}

/// Evaluate a leaf position with the NN, returning `(value, child_priors)` in one pass.
fn nn_leaf_eval(
    net: &HextoeNet,
    device: &Device,
    state: &GameState,
    root_player: Player,
    rng: &mut impl Rng,
) -> (f32, Option<Vec<(Pos, f32)>>) {
    if state.is_terminal() {
        let v = match state.winner {
            Some(p) if p == root_player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        return (v, None);
    }
    let center = board_center(state);
    match net.evaluate_state(state, device) {
        Ok((probs, v)) => {
            let cp = state.current_player();
            let value = if cp == root_player { v } else { -v };
            let legal = state.legal_actions();
            let priors: Vec<(Pos, f32)> = legal
                .iter()
                .filter_map(|&pos| {
                    let idx = action_to_index(pos, center)?;
                    Some((pos, probs[idx]))
                })
                .collect();
            (value, Some(priors))
        }
        Err(_) => {
            let (v, _) = RandomRollout.rollout(state.clone(), root_player, rng);
            (v, None)
        }
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
    /// Single NN forward pass: returns leaf value + policy priors for the leaf's children.
    fn rollout(&self, state: GameState, root_player: Player, rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>) {
        nn_leaf_eval(self.net, self.device, &state, root_player, rng)
    }

    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        nn_priors(self.net, self.device, state)
    }

    const PARALLEL_SAFE: bool = false;
}

/// MCTS rollout policy for new-vs-best evaluation: the player to move uses their own net.
pub struct DualNetRollout<'a> {
    pub new_net: &'a HextoeNet,
    pub best_net: &'a HextoeNet,
    pub new_player: Player,
    pub device: &'a Device,
}

impl RolloutPolicy for DualNetRollout<'_> {
    fn rollout(&self, state: GameState, root_player: Player, rng: &mut impl Rng) -> (f32, Option<Vec<(Pos, f32)>>) {
        let net = if state.current_player() == self.new_player {
            self.new_net
        } else {
            self.best_net
        };
        nn_leaf_eval(net, self.device, &state, root_player, rng)
    }

    fn priors_only(&self, state: &GameState) -> Option<Vec<(Pos, f32)>> {
        let net = if state.current_player() == self.new_player {
            self.new_net
        } else {
            self.best_net
        };
        nn_priors(net, self.device, state)
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
