#!/usr/bin/env python3
"""
Hextoe CNN pre-trainer — PyTorch/MPS edition.

Usage:
    python train.py games*.json [--epochs 200] [--batch 128] [--lr 3e-4] [--out hextoe_model_best.safetensors]

Module names are chosen to exactly match candle's VarBuilder.pp() key paths so
weights saved here load directly into the Rust inference code.

CTRL+C → saves checkpoint and exits cleanly.
"""

import argparse
import glob
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
except ImportError:
    print("ERROR: safetensors not installed.  Run: pip install safetensors")
    sys.exit(1)

try:
    import hextoe_py
except ImportError:
    print("ERROR: hextoe_py not built.  Run: cd hextoe-py && maturin develop --release")
    sys.exit(1)

GRID = hextoe_py.GRID          # 33
CHANNELS = hextoe_py.CHANNELS  # 4
HIDDEN = 64
RES_BLOCKS = 4

# ── Model (names match candle VarBuilder keys) ────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        return F.relu(out + x)


class HextoeNet(nn.Module):
    """
    Architecture matching nn.rs exactly.
    Attribute names == candle VarBuilder.pp() names so safetensors keys are identical.
    """
    def __init__(self):
        super().__init__()
        H, G = HIDDEN, GRID

        # Trunk — "init" / "init_bn" / "r0".."r{N-1}"
        self.init = nn.Conv2d(CHANNELS, H, 3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(H)
        for i in range(RES_BLOCKS):
            setattr(self, f"r{i}", ResBlock(H))

        # Policy head — "pc" / "p_bn" / "pf"
        self.pc = nn.Conv2d(H, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.pf = nn.Linear(2 * G * G, G * G)

        # Value head — "vc" / "v_bn" / "vf1" / "vf2"
        self.vc = nn.Conv2d(H, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.vf1 = nn.Linear(G * G, 256)
        self.vf2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
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


# ── safetensors ↔ PyTorch weight bridge ──────────────────────────────────────

def _candle_key(pt_key: str) -> str:
    """PyTorch state_dict key → candle safetensors key (they're identical here)."""
    return pt_key


def save_weights(model: HextoeNet, path: str):
    """Save to safetensors, skipping num_batches_tracked (candle doesn't use it)."""
    sd = {
        k: v.contiguous().cpu()
        for k, v in model.state_dict().items()
        if "num_batches_tracked" not in k
    }
    st_save(sd, path)


def load_weights(model: HextoeNet, path: str) -> bool:
    """Load from safetensors (candle or Python checkpoint). Returns True on success."""
    if not Path(path).exists():
        return False
    try:
        sd = st_load(path, device="cpu")
        # Allow missing keys (e.g. num_batches_tracked) and unexpected keys.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  Warning: unexpected keys in {path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        return True
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return False


# ── Training ──────────────────────────────────────────────────────────────────

def train_step(model, states, policies, values, opt, device):
    states = torch.from_numpy(np.array(states)).to(device)
    policies = torch.from_numpy(np.array(policies)).to(device)
    values = torch.from_numpy(np.array(values)).unsqueeze(1).to(device)

    policy_logits, value_out = model(states)

    log_p = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policies * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value_out, values)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description="Hextoe CNN supervised pre-trainer")
    ap.add_argument("games", nargs="+", help="JSON game file(s) — shell globs are expanded")
    ap.add_argument("--epochs",   type=int,   default=200)
    ap.add_argument("--batch",    type=int,   default=128)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--out",      default="hextoe_model_best.safetensors")
    ap.add_argument("--latest",   default="hextoe_model_latest.safetensors")
    ap.add_argument("--cpu",      action="store_true")
    args = ap.parse_args()

    # Expand any shell globs that the shell didn't expand (e.g. on Windows).
    paths = []
    for p in args.games:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Loading {len(paths)} game file(s)…")
    sampler = hextoe_py.Sampler(paths)
    n_samples = len(sampler)
    steps_per_epoch = max(1, n_samples // args.batch)
    total_steps = args.epochs * steps_per_epoch
    print(f"{n_samples} positions  →  {steps_per_epoch} steps/epoch  ×  {args.epochs} epochs  =  {total_steps} total steps")

    # ── Model ────────────────────────────────────────────────────────────────
    device = pick_device(args.cpu)
    print(f"Device: {device}")

    model = HextoeNet().to(device)

    loaded = False
    for ckpt in [args.out, args.latest, "hextoe_model.safetensors"]:
        if load_weights(model, ckpt):
            print(f"Loaded weights from {ckpt}")
            loaded = True
            break
    if not loaded:
        print("No existing checkpoint — training from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── CTRL+C handler ───────────────────────────────────────────────────────
    interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal interrupted
        if interrupted:
            print("\nForce-quitting.")
            sys.exit(1)
        print("\nCTRL+C — will save after this batch. Press again to force-quit.")
        interrupted = True
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"Checkpointing every epoch to {args.latest}  |  CTRL+C to save and exit")
    print("─" * 80)

    t0 = time.time()
    progress_every = max(1, steps_per_epoch // 10)  # ~10% increments

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = epoch_pol = epoch_val = 0.0
        epoch_t0 = time.time()

        for step in range(1, steps_per_epoch + 1):
            states, policies, values = sampler.sample_batch(args.batch)
            loss, pol_loss, val_loss = train_step(model, states, policies, values, opt, device)
            epoch_loss += loss
            epoch_pol  += pol_loss
            epoch_val  += val_loss

            if step % progress_every == 0 or step == steps_per_epoch:
                steps_done = (epoch - 1) * steps_per_epoch + step
                elapsed = time.time() - t0
                sps = steps_done / elapsed
                eta = (total_steps - steps_done) / sps if sps > 0 else 0
                mean = epoch_loss / step
                print(
                    f"epoch {epoch:>4}/{args.epochs}  "
                    f"step {step:>5}/{steps_per_epoch}  "
                    f"loss {mean:.4f} (pol {epoch_pol/step:.4f} val {epoch_val/step:.4f})  "
                    f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                    flush=True,
                )

            if interrupted:
                break

        # Save every epoch.
        epoch_secs = time.time() - epoch_t0
        mean = epoch_loss / steps_per_epoch
        save_weights(model, args.latest)
        print(f"── epoch {epoch}/{args.epochs}  loss {mean:.4f}  ({epoch_secs:.1f}s)  → {args.latest}", flush=True)

        if interrupted:
            break

    # Final save.
    print("─" * 80)
    for path in set([args.out, args.latest]):
        save_weights(model, path)
        print(f"Saved → {path}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
