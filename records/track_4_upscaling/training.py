from __future__ import annotations

import contextlib
import csv
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .model import GPT, GPTConfig, build_optimizers, non_embedding_params


# Training hyperparameters (versioned). These are fixed by the benchmark and
# identical for every run. Submissions do not change them. The defaults are the
# real 8xH100 values; tiny CPU toy runs override them (see train_fineweb_demo.py).

@dataclass
class Hyperparameters:
    version: str = "H1"
    batch_seqs: int = 32
    seq_len: int = 1024
    embed_lr: float = 0.3
    head_lr: float = 0.004
    scalar_lr: float = 0.02
    muon_lr: float = 0.02
    muon_wd: float = 0.0
    cooldown_frac: float = 0.4
    warmup_steps: int = 200

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_HPARAMS = Hyperparameters()


def lr_scale(step: int, total_steps: int, cooldown_frac: float, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = step / max(1, total_steps)
    if progress < 1 - cooldown_frac:
        return 1.0
    return max(0.0, (1 - progress) / cooldown_frac)


def _autocast(amp_dtype: torch.dtype | None, on_cuda: bool):
    if amp_dtype is not None and on_cuda:
        return torch.autocast("cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: GPT, val_batches, *, amp_dtype: torch.dtype | None = None) -> float:
    """Mean validation loss over a fixed list of batches. Under torch.distributed
    the per-rank loss and token sums are reduced so the result is the global mean
    over the whole (rank-sharded) validation set."""
    model.eval()
    loss_sum, tokens = 0.0, 0
    device = None
    for inputs, targets in val_batches:
        device = inputs.device
        with _autocast(amp_dtype, inputs.is_cuda):
            loss_sum += float(model(inputs, targets))
        tokens += targets.numel()
    model.train()
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([loss_sum, float(tokens)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum, tokens = float(t[0]), int(t[1].item())
    return loss_sum / max(1, tokens)


@dataclass
class TrainResult:
    steps: int
    tokens: int
    final_loss: float
    reached_target: bool
    flops: float
    loss_history: list  # list[(step, val_loss)]
    wall_seconds: float = 0.0


def train_model(
    model: GPT,
    train_gen,
    val_batches,
    *,
    hparams: Hyperparameters,
    max_steps: int,
    schedule_steps: int | None = None,
    target_loss: float | None = None,
    eval_every: int = 25,
    log: bool = False,
    amp_dtype: torch.dtype | None = None,
    grad_sync=None,
    flops_per_step: float,
    world_size: int = 1,
) -> TrainResult:
    """Train until `max_steps` or until val loss <= `target_loss`.

    Cost is counted from the START of this call. ``flops_per_step`` is the
    measured FLOPs of one training step (see model.measure_step_flops); the run
    total is that times the steps taken. Prior cost (e.g. the small-model run) is
    accounted for by the caller.

    ``schedule_steps`` is the LR-schedule horizon (warmup/stable/cooldown are
    sized to it). It defaults to ``max_steps``; the benchmark sets it to the fixed
    realistic budget so every run shares one schedule and L2 is reached in the
    stable phase. ``max_steps`` only bounds the loop.
    """
    sched_steps = schedule_steps if schedule_steps is not None else max_steps
    optimizers = build_optimizers(
        model, embed_lr=hparams.embed_lr, head_lr=hparams.head_lr,
        scalar_lr=hparams.scalar_lr, muon_lr=hparams.muon_lr, muon_wd=hparams.muon_wd,
    )
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    warmup = hparams.warmup_steps
    tokens_per_step = hparams.batch_seqs * hparams.seq_len * world_size  # global tokens/step

    history: list = []
    steps_done = 0
    reached = False
    final_loss = float("nan")
    on_cuda = val_batches[0][0].is_cuda
    t0 = time.time()
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps:
            vl = evaluate(model, val_batches, amp_dtype=amp_dtype)
            history.append((step, vl))
            final_loss = vl
            if log:
                print(f"    step {step:4d}/{max_steps}  val_loss {vl:.4f}")
            if target_loss is not None and vl <= target_loss:
                reached = True
                steps_done = step
                break
        if step == max_steps:
            steps_done = step
            break

        inputs, targets = next(train_gen)
        with _autocast(amp_dtype, on_cuda):
            loss = model(inputs, targets)
        (loss / targets.numel()).backward()
        if grad_sync is not None:
            grad_sync(model)
        scale = lr_scale(step, sched_steps, hparams.cooldown_frac, warmup)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
            opt.step()
        model.zero_grad(set_to_none=True)
        steps_done = step + 1

    tokens = steps_done * tokens_per_step
    flops = flops_per_step * steps_done
    return TrainResult(
        steps=steps_done,
        tokens=tokens,
        final_loss=final_loss,
        reached_target=reached,
        flops=flops,
        loss_history=history,
        wall_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile wraps the module, so its state_dict keys carry an
    ``_orig_mod.`` prefix. Strip it so checkpoints are framework-agnostic."""
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def save_checkpoint(path: str, model: GPT, *, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    config = getattr(model, "_orig_mod", model).config
    torch.save({
        "state_dict": _strip_compile_prefix(model.state_dict()),
        "config": config.to_dict(),
        "meta": meta or {},
    }, path)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(_strip_compile_prefix(ckpt["state_dict"]))
    model.to(device)
    return model, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CSV helpers (shared with train_upscaled_gpt.py)
# ---------------------------------------------------------------------------

def append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


REFERENCE_FIELDS = [
    "ref_version", "hparams_version", "kind", "params_nonembed",
    "L1", "L2", "steps_to_L2", "tokens_to_L2", "flops_to_L2", "wall_seconds", "reached", "seed",
]


def write_reference(path: str, *, ref_version: str, kind: str, model: GPT,
                    L1, L2, result: TrainResult, seed: int,
                    hparams_version: str = DEFAULT_HPARAMS.version) -> None:
    append_csv(path, {
        "ref_version": ref_version,
        "hparams_version": hparams_version,
        "kind": kind,  # "from_scratch_large"
        "params_nonembed": non_embedding_params(model),
        "L1": L1, "L2": L2,
        "steps_to_L2": result.steps,
        "tokens_to_L2": result.tokens,
        "flops_to_L2": result.flops,
        "wall_seconds": round(result.wall_seconds, 1),
        "reached": int(result.reached_target),
        "seed": seed,
    }, REFERENCE_FIELDS)
