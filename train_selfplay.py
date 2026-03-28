#!/usr/bin/env python3
"""
Hextoe self-play trainer (AlphaZero style) — PyTorch/MPS edition.

Each iteration:
  1. Generate games (mix of naive-play and self-play vs current champion).
  2. Add positions to a rolling replay buffer.
  3. Train for a fixed number of steps.
  4. Evaluate candidate vs champion (win-rate).  Promote only on improvement.

Usage:
    python train_selfplay.py [--iters 200] [--games-per-iter 50]
                             [--sims 50] [--naive-frac 0.3]
                             [--batch 256] [--train-steps 200]
                             [--buffer 30000] [--eval-games 20]
                             [--promote-threshold 0.55]
                             [--champion hextoe_model_best.safetensors]
                             [--latest   hextoe_model_latest.safetensors]
                             [--cpu]
"""

import argparse
import collections
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load, save_file as st_save
from pathlib import Path

try:
    import hextoe_py
except ImportError:
    print("ERROR: hextoe_py not built.  Run: cd hextoe-py && maturin develop --release")
    raise

GRID = hextoe_py.GRID
CHANNELS = hextoe_py.CHANNELS
G2 = GRID * GRID
HIDDEN = 64
RES_BLOCKS = 4

# ── Model (identical to train.py) ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        return F.relu(out + x)


class HextoeNet(nn.Module):
    def __init__(self):
        super().__init__()
        H, G = HIDDEN, GRID
        self.init = nn.Conv2d(CHANNELS, H, 3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(H)
        for i in range(RES_BLOCKS):
            setattr(self, f"r{i}", ResBlock(H))
        self.pc = nn.Conv2d(H, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.pf = nn.Linear(2 * G * G, G * G)
        self.vc = nn.Conv2d(H, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.vf1 = nn.Linear(G * G, 256)
        self.vf2 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.init_bn(self.init(x)))
        for i in range(RES_BLOCKS):
            h = getattr(self, f"r{i}")(h)
        p = F.relu(self.p_bn(self.pc(h)))
        p = p.view(p.size(0), -1)
        policy = self.pf(p)
        v = F.relu(self.v_bn(self.vc(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.vf1(v))
        value = torch.tanh(self.vf2(v))
        return policy, value


# ── Weight I/O ────────────────────────────────────────────────────────────────

def save_weights(model, path):
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()
          if "num_batches_tracked" not in k}
    st_save(sd, path)


def load_weights(model, path):
    if not Path(path).exists():
        return False
    try:
        sd = st_load(path, device="cpu")
        model.load_state_dict(sd, strict=False)
        return True
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return False


def model_copy(src, device):
    """Deep copy of a model (used for champion snapshot)."""
    dst = HextoeNet().to(device)
    dst.load_state_dict(src.state_dict())
    dst.eval()
    return dst


# ── MCTS ──────────────────────────────────────────────────────────────────────

def net_eval(model, state, device):
    """Run the network on a single state; returns (policy_probs, value)."""
    enc = state.encode()  # [C, H, W] numpy
    x = torch.from_numpy(enc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, v = model(x)
    probs = F.softmax(logits[0], dim=0).cpu().numpy()
    return probs, float(v[0, 0].cpu())


def net_eval_batch(model, nodes, device):
    """Evaluate a list of MCTSNodes in one forward pass. Returns (probs_list, values_list)."""
    encs = np.stack([n.state.encode() for n in nodes])
    x = torch.from_numpy(encs).to(device)
    with torch.no_grad():
        logits, v = model(x)
    probs_list = F.softmax(logits, dim=1).cpu().numpy()
    values_list = v[:, 0].cpu().numpy()
    return probs_list, values_list


class MCTSNode:
    __slots__ = ("state", "prior", "visit", "value_sum", "children", "expanded")

    def __init__(self, state, prior=1.0):
        self.state = state
        self.prior = prior
        self.visit = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode
        self.expanded = False

    def q(self):
        return self.value_sum / self.visit if self.visit > 0 else 0.0

    def ucb(self, parent_visit, c_puct, same_player=False):
        # If same player moves again (pair-move rule), child's q is already from our
        # perspective — use it directly.  If the player changed, negate to convert from
        # the opponent's perspective to ours.
        q = self.q() if same_player else -self.q()
        return q + c_puct * self.prior * (parent_visit ** 0.5) / (1 + self.visit)


VIRTUAL_LOSS = 2.0  # temporarily penalise in-flight nodes to spread parallel selection


def _select_leaf(root, c_puct):
    """Walk tree to a leaf, applying virtual loss along the way. Returns (node, path)."""
    node = root
    path = [node]
    while node.expanded and not node.state.is_terminal():
        parent_player = node.state.current_player()
        best_child = max(
            node.children.values(),
            key=lambda ch: ch.ucb(
                node.visit + 1, c_puct,
                same_player=(ch.state.current_player() == parent_player),
            ),
        )
        # Apply virtual loss so parallel selections spread across different branches.
        best_child.visit += 1
        best_child.value_sum -= VIRTUAL_LOSS
        node = best_child
        path.append(node)
    return node, path


def _backup(path, leaf_value):
    """Undo virtual loss and backup real value."""
    value = leaf_value
    for i in range(len(path) - 1, -1, -1):
        n = path[i]
        # Undo the virtual-loss visit that was pre-applied during selection
        # (only for non-root nodes that had it applied).
        if i > 0:
            n.value_sum += VIRTUAL_LOSS  # cancel virtual loss
            # visit count was already incremented; just update value_sum net
        else:
            n.visit += 1
            n.value_sum += value
            break
        n.value_sum += value
        if path[i].state.current_player() != path[i - 1].state.current_player():
            value = -value


def mcts_search(model, root_state, n_sims, device, c_puct=2.0, temperature=1.0, eval_batch=8):
    """
    Run MCTS from root_state with batched leaf evaluation.
    eval_batch leaves are selected per round and evaluated in one forward pass,
    giving much better GPU utilisation than batch=1.
    """
    root = MCTSNode(root_state.clone())
    _expand(root, model, device)  # prime root priors (single eval)

    sims_done = 0
    while sims_done < n_sims:
        batch = min(eval_batch, n_sims - sims_done)

        # Select `batch` leaves (with virtual loss so they diverge).
        leaves, paths = [], []
        terminal_backups = []
        for _ in range(batch):
            leaf, path = _select_leaf(root, c_puct)
            if leaf.state.is_terminal():
                terminal_backups.append(path)
                leaves.append(None)
            else:
                leaves.append(leaf)
            paths.append(path)

        # Batch-evaluate all non-terminal leaves in one network call.
        non_term = [(i, leaf) for i, leaf in enumerate(leaves) if leaf is not None]
        if non_term:
            idxs, nodes = zip(*non_term)
            probs_batch, values_batch = net_eval_batch(model, list(nodes), device)
            for j, (orig_i, leaf) in enumerate(non_term):
                if not leaf.expanded:
                    _expand_with_probs(leaf, probs_batch[j])
                _backup(paths[orig_i], float(values_batch[j]))

        for path in terminal_backups:
            _backup(path, -1.0)

        sims_done += batch

    # Build policy from visit counts.
    pi = np.zeros(G2, dtype=np.float32)
    for action, child in root.children.items():
        idx = root_state.action_to_index(action[0], action[1])
        if 0 <= idx < G2:
            pi[idx] = child.visit

    if pi.sum() == 0:
        pi[:] = 1.0 / G2
    elif temperature < 0.1:
        best = np.argmax(pi)
        pi[:] = 0.0
        pi[best] = 1.0
    else:
        pi = pi ** (1.0 / temperature)
        s = pi.sum()
        pi[:] = 1.0 / G2 if s == 0 else pi / s

    # Root Q is a much better value target than sparse game outcome: it averages
    # over n_sims explored paths with the current network, giving a per-position
    # estimate that already accounts for the opponent's best responses.
    root_q = root.q()
    return pi, root_q


def _expand(node, model, device):
    """Expand node by evaluating with network (used for root priming)."""
    if node.expanded:
        return
    actions = node.state.legal_actions()
    if not actions:
        node.expanded = True
        return
    probs, _ = net_eval(model, node.state, device)
    _expand_with_probs(node, probs)  # sets node.expanded = True


TACTICAL_BLEND = 0.5  # weight given to tactical priors when threats exist (0 = NN only, 1 = tactical only)

def _expand_with_probs(node, probs):
    """Expand node using NN policy probs, blended with tactical heuristics when threats exist."""
    if node.expanded:
        return
    node.expanded = True
    actions = node.state.legal_actions()
    if not actions:
        return

    # Get tactical priors from Rust (empty list when position is quiet).
    tac_list = node.state.tactical_priors()
    tac = dict(tac_list) if tac_list else {}

    for action in actions:
        idx = node.state.action_to_index(action[0], action[1])
        nn_prior = float(probs[idx]) if 0 <= idx < G2 else 1.0 / len(actions)
        if tac:
            tac_prior = tac.get(action, 0.0)
            prior = (1.0 - TACTICAL_BLEND) * nn_prior + TACTICAL_BLEND * tac_prior
        else:
            prior = nn_prior
        child_state = node.state.clone()
        child_state.place(action[0], action[1])
        node.children[action] = MCTSNode(child_state, prior=prior)


def pick_action_from_pi(state, pi):
    """Sample an action from a policy vector, restricted to legal moves."""
    actions = state.legal_actions()
    center = state.board_center()
    weights = []
    for a in actions:
        idx = state.action_to_index(a[0], a[1])
        w = float(pi[idx]) if 0 <= idx < G2 else 0.0
        weights.append(max(w, 1e-8))
    total = sum(weights)
    r = random.random() * total
    cumsum = 0.0
    for a, w in zip(actions, weights):
        cumsum += w
        if cumsum >= r:
            return a
    return actions[-1]


# ── Game generation ───────────────────────────────────────────────────────────

def play_game_self(model, n_sims, device, temp_moves=10, max_moves=120, eval_batch=8):
    """Self-play game. Returns list of (state_enc, pi, outcome_placeholder)."""
    state = hextoe_py.PyGameState()
    records = []  # (enc [C,H,W], pi [G²], player_at_move)

    move_num = 0
    while not state.is_terminal() and move_num < max_moves:
        temp = 1.0 if move_num < temp_moves else 0.05
        pi, mcts_q = mcts_search(model, state, n_sims, device, temperature=temp, eval_batch=eval_batch)
        enc = state.encode()
        player = state.current_player()
        records.append((enc, pi, player, mcts_q))

        action = pick_action_from_pi(state, pi)
        state.place(action[0], action[1])
        move_num += 1

    winner = state.winner()
    samples = []
    for enc, pi, player, mcts_q in records:
        if winner == -1:
            game_outcome = 0.0
        elif winner == player:
            game_outcome = 1.0
        else:
            game_outcome = -1.0
        # Blend MCTS root-Q with sparse game outcome.
        # MCTS Q is a richer, per-position signal; game outcome anchors it to reality.
        value_target = 0.25 * mcts_q + 0.75 * game_outcome
        samples.append((enc, pi, value_target))
    return samples


def play_game_vs_naive(model, n_sims, device, model_plays_x=True, max_moves=120, eval_batch=8):
    """
    One game: model (X or O) vs naive opponent.
    Returns list of (enc, pi, outcome) for model's moves only.
    """
    state = hextoe_py.PyGameState()
    model_player = 0 if model_plays_x else 1
    records = []
    move_num = 0

    while not state.is_terminal() and move_num < max_moves:
        current = state.current_player()
        if current == model_player:
            pi, mcts_q = mcts_search(model, state, n_sims, device, temperature=1.0, eval_batch=eval_batch)
            enc = state.encode()
            records.append((enc, pi, model_player, mcts_q))
            action = pick_action_from_pi(state, pi)
        else:
            action = state.naive_move()
            if action is None:
                break
        state.place(action[0], action[1])
        move_num += 1

    winner = state.winner()
    samples = []
    for enc, pi, player, mcts_q in records:
        if winner == -1:
            game_outcome = 0.0
        elif winner == player:
            game_outcome = 1.0
        else:
            game_outcome = -1.0
        value_target = 0.25 * mcts_q + 0.75 * game_outcome
        samples.append((enc, pi, value_target))
    return samples


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(model, batch, opt, device):
    encs, pis, outcomes = zip(*batch)
    states = torch.from_numpy(np.stack(encs)).to(device)
    policies = torch.from_numpy(np.stack(pis)).to(device)
    values = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    logits, value_out = model(states)
    log_p = F.log_softmax(logits, dim=1)
    policy_loss = -(policies * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value_out, values)
    loss = policy_loss + 1.0 * value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), policy_loss.item(), value_loss.item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(candidate, champion, n_games, n_sims, device, it=0, max_moves=120, eval_batch=8):
    """
    Play `n_games` between candidate (X half the time) and champion.
    Returns candidate win-rate in [0, 1].
    """
    candidate.eval()
    champion.eval()
    wins = 0
    for g in range(n_games):
        t_g = time.time()
        cand_is_x = (g % 2 == 0)
        state = hextoe_py.PyGameState()
        move_num = 0
        while not state.is_terminal() and move_num < max_moves:
            current = state.current_player()
            is_cand = (cand_is_x and current == 0) or (not cand_is_x and current == 1)
            model = candidate if is_cand else champion
            pi, _ = mcts_search(model, state, n_sims, device, temperature=0.0, eval_batch=eval_batch)
            action = pick_action_from_pi(state, pi)
            state.place(action[0], action[1])
            move_num += 1
        winner = state.winner()
        cand_player = 0 if cand_is_x else 1
        if winner == cand_player:
            wins += 1
        print(f"  iter {it} eval  {g+1}/{n_games}  wins {wins}/{g+1}  {time.time()-t_g:.1f}s", flush=True)
    return wins / n_games


# ── Main ──────────────────────────────────────────────────────────────────────

def pick_device(force_cpu):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters",              type=int,   default=200)
    ap.add_argument("--games-per-iter",     type=int,   default=20)
    ap.add_argument("--sims",               type=int,   default=25,
                    help="MCTS simulations per move")
    ap.add_argument("--naive-frac",         type=float, default=0.3,
                    help="Fraction of games played vs naive (rest are self-play)")
    ap.add_argument("--lr",                 type=float, default=3e-4)
    ap.add_argument("--mcts-batch",         type=int,   default=8,
                    help="Leaves evaluated per network call inside MCTS (higher = faster GPU)")
    ap.add_argument("--max-moves",          type=int,   default=300,
                    help="Terminate games after this many moves (prevents runaway games)")
    ap.add_argument("--eval-every",         type=int,   default=5,
                    help="Run champion evaluation every N iterations")
    ap.add_argument("--batch",              type=int,   default=256)
    ap.add_argument("--train-steps",        type=int,   default=200)
    ap.add_argument("--buffer",             type=int,   default=30000,
                    help="Max replay buffer size")
    ap.add_argument("--eval-games",         type=int,   default=10)
    ap.add_argument("--promote-threshold",  type=float, default=0.50,
                    help="Win-rate vs champion required to promote")
    ap.add_argument("--champion", default="hextoe_model_best.safetensors")
    ap.add_argument("--latest",   default="hextoe_model_latest.safetensors")
    ap.add_argument("--cpu",      action="store_true")
    args = ap.parse_args()

    device = pick_device(args.cpu)
    print(f"Device: {device}")

    # Load or init candidate model.
    candidate = HextoeNet().to(device)
    loaded = False
    for ckpt in [args.champion, args.latest, "hextoe_model.safetensors"]:
        if load_weights(candidate, ckpt):
            print(f"Loaded candidate from {ckpt}")
            loaded = True
            break
    if not loaded:
        print("Starting from scratch")

    # Champion is a frozen copy — promote only when candidate improves.
    champion = model_copy(candidate, device)

    opt = torch.optim.AdamW(candidate.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = collections.deque(maxlen=args.buffer)

    interrupted = False
    def _sigint(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        print("\nCTRL+C — finishing iteration then saving. Press again to force-quit.")
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    t0 = time.time()
    print(f"Self-play training: {args.iters} iters × {args.games_per_iter} games, "
          f"{args.sims} sims/move, promote@{args.promote_threshold:.0%}")
    print("─" * 80)

    for it in range(1, args.iters + 1):
        it_t0 = time.time()
        candidate.eval()

        # 1. Generate games.
        #n_naive = max(1, int(args.games_per_iter * args.naive_frac))
        n_naive = int(args.games_per_iter * args.naive_frac)
        n_self  = args.games_per_iter - n_naive
        new_samples = 0

        total_games = n_naive + n_self
        for g in range(n_naive):
            t_g = time.time()
            samples = play_game_vs_naive(candidate, args.sims, device,
                                         model_plays_x=(g % 2 == 0),
                                         max_moves=args.max_moves,
                                         eval_batch=args.mcts_batch)
            replay.extend(samples)
            new_samples += len(samples)
            print(f"  iter {it} game {g+1}/{total_games} (vs-naive) "
                  f"{len(samples)} positions  {time.time()-t_g:.1f}s", flush=True)

        for gi in range(n_self):
            t_g = time.time()
            samples = play_game_self(candidate, args.sims, device,
                                     max_moves=args.max_moves,
                                     eval_batch=args.mcts_batch)
            replay.extend(samples)
            new_samples += len(samples)
            print(f"  iter {it} game {n_naive+gi+1}/{total_games} (self-play) "
                  f"{len(samples)} positions  {time.time()-t_g:.1f}s", flush=True)

        # 2. Train.
        candidate.train()
        total_loss = total_pol = total_val = 0.0
        effective_steps = min(args.train_steps, len(replay) // args.batch)
        effective_steps = max(effective_steps, 1)

        for _ in range(effective_steps):
            batch = random.sample(replay, min(args.batch, len(replay)))
            loss, pol, val = train_step(candidate, batch, opt, device)
            total_loss += loss
            total_pol  += pol
            total_val  += val

        mean_loss = total_loss / effective_steps
        mean_pol  = total_pol  / effective_steps
        mean_val  = total_val  / effective_steps

        # 3. Save latest.
        save_weights(candidate, args.latest)

        # 4. Evaluate and possibly promote champion (every eval_every iters).
        candidate.eval()
        if it % args.eval_every == 0:
            win_rate = evaluate(candidate, champion, args.eval_games, args.sims, device,
                                it=it, max_moves=args.max_moves, eval_batch=args.mcts_batch)
        else:
            win_rate = -1.0  # skipped
        promoted = ""
        if win_rate < 0:
            promoted = "  (eval skipped)"
        elif win_rate >= args.promote_threshold:
            champion = model_copy(candidate, device)
            save_weights(candidate, args.champion)
            promoted = "  *** PROMOTED (champion updated) ***"
        elif win_rate >= 0:
            promoted = "  (no promote — continuing)"

        elapsed = time.time() - t0
        it_secs = time.time() - it_t0
        print(
            f"iter {it:>4}/{args.iters}  "
            f"buf {len(replay):>6}  new {new_samples:>4}  "
            f"loss {mean_loss:.4f} (pol {mean_pol:.4f} val {mean_val:.4f})  "
            f"winrate {win_rate:.1%}  " if win_rate >= 0 else "winrate —       "
            f"{it_secs:.0f}s  elapsed {elapsed:.0f}s"
            f"{promoted}",
            flush=True,
        )

        if interrupted:
            break

    print("─" * 80)
    save_weights(candidate, args.latest)
    print(f"Saved latest → {args.latest}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
