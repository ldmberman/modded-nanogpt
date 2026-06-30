"""Run the upscaling benchmark.

Single process:
    python -m records.track_4_upscaling.run small    --l1 3.6
    python -m records.track_4_upscaling.run refs    --l1 3.6 --l2 3.4
    python -m records.track_4_upscaling.run upscale --l1 3.6 --l2 3.4 --op ligo

Multi-GPU (one node, 8 ranks):
    torchrun --standalone --nproc_per_node=8 -m records.track_4_upscaling.run refs --l1 3.6 --l2 3.4
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from .model import (GPT, GPTConfig, build_optimizers, init_params, set_seed,
                    measure_step_flops, fineweb_data_generator, fineweb_val_batches)
from .training import (train_model, save_checkpoint, load_checkpoint,
                         write_reference)
from .train_upscaled_gpt import run_upscale
from . import config as C


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_dist():
    """Return (rank, world_size, local_rank, device). Works with or without torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            return rank, world, local, torch.device(f"cuda:{local}")
        return rank, world, local, torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def make_grad_sync(world_size: int):
    """Average gradients across ranks before the optimizer step.
    Uses SUM-then-divide rather than ReduceOp.AVG so it works on gloo (CPU smoke)
    as well as nccl; AVG is nccl-only."""
    if world_size <= 1:
        return None

    def sync(module):
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

    return sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_train_gen(device, rank, world):
    return fineweb_data_generator(C.TRAIN_GLOB, C.HPARAMS.batch_seqs,
                                  C.HPARAMS.seq_len, device, rank=rank, world_size=world)


def build_val_batches(device, rank, world):
    return fineweb_val_batches(C.VAL_GLOB, C.HPARAMS.batch_seqs, C.HPARAMS.seq_len,
                               device, total_tokens=C.VAL_TOKENS, rank=rank, world_size=world)


def measured_flops_per_step(config: GPTConfig, device, rank, world) -> float:
    """One global training step's measured FLOPs (fwd + bwd + optimizer), summed
    across ranks so it matches the analytic global cost.

    Per-step FLOPs are exactly affine in the number of sequences: fwd+bwd scale
    linearly while the optimizer (Muon Newton-Schulz) is batch-independent. We
    therefore measure two small micro-batches and extrapolate to the full batch,
    which avoids running a full-batch fwd+bwd (too memory-heavy for the bigger
    model on a single GPU) while staying exact."""
    probe = GPT(config).to(device)
    init_params(probe)
    opts = build_optimizers(
        probe, embed_lr=C.HPARAMS.embed_lr, head_lr=C.HPARAMS.head_lr,
        scalar_lr=C.HPARAMS.scalar_lr, muon_lr=C.HPARAMS.muon_lr,
        muon_wd=C.HPARAMS.muon_wd,
    )
    inputs, targets = next(new_train_gen(device, rank, world))
    full_b = C.HPARAMS.batch_seqs

    def at(nb: int) -> float:
        mb = (inputs[:nb].contiguous(), targets[:nb].contiguous())
        return float(measure_step_flops(probe, mb, optimizers=opts, autocast_dtype=C.AMP_DTYPE))

    b1, b2 = 2, 4
    f1, f2 = at(b1), at(b2)
    per_seq = (f2 - f1) / (b2 - b1)
    const = f2 - per_seq * b2
    local = per_seq * full_b + const
    if world > 1:
        t = torch.tensor([float(local)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    return float(local)


def maybe_compile(model):
    if C.COMPILE and torch.cuda.is_available():
        return torch.compile(model)
    return model


def build_model(config: GPTConfig, device, seed: int) -> GPT:
    set_seed(seed)
    model = GPT(config).to(device)
    init_params(model)
    return model


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_small(args, ctx):
    rank, world, device, is_main = ctx
    flops_per_step = measured_flops_per_step(C.SMALL_CONFIG, device, rank, world)
    model = maybe_compile(build_model(C.SMALL_CONFIG, device, seed=1))
    res = train_model(model, new_train_gen(device, rank, world), build_val_batches(device, rank, world),
                      hparams=C.HPARAMS, max_steps=args.max_steps,
                      schedule_steps=C.SMALL_BUDGET_STEPS, target_loss=args.l1,
                      eval_every=args.eval_every, log=is_main,
                      amp_dtype=C.AMP_DTYPE, grad_sync=make_grad_sync(world),
                      flops_per_step=flops_per_step, world_size=world)
    if is_main:
        save_checkpoint(args.ckpt, model, meta={"L1": args.l1, "flops": res.flops})
        print(f"[small] reached={res.reached_target} steps={res.steps} "
              f"loss={res.final_loss:.4f} flops={res.flops:.4g} -> {args.ckpt}")


def phase_refs(args, ctx):
    rank, world, device, is_main = ctx
    sync = make_grad_sync(world)
    flops_per_step_big = measured_flops_per_step(C.BIG_CONFIG, device, rank, world)

    val = build_val_batches(device, rank, world)
    big = maybe_compile(build_model(C.BIG_CONFIG, device, seed=2))
    r_fs = train_model(big, new_train_gen(device, rank, world), val,
                       hparams=C.HPARAMS, max_steps=args.max_steps,
                       schedule_steps=C.BIG_BUDGET_STEPS, target_loss=args.l2,
                       eval_every=args.eval_every, log=is_main,
                       amp_dtype=C.AMP_DTYPE, grad_sync=sync, flops_per_step=flops_per_step_big,
                       world_size=world)
    if is_main:
        write_reference(args.refs, ref_version=args.ref_version, kind="from_scratch_large",
                        model=big, L1=args.l1, L2=args.l2, result=r_fs, seed=2,
                        hparams_version=C.HPARAMS.version)
        print(f"[refs] from_scratch_large reached={r_fs.reached_target} "
              f"steps={r_fs.steps} flops={r_fs.flops:.4g}")


def phase_upscale(args, ctx):
    rank, world, device, is_main = ctx
    hp = C.HPARAMS  # fixed schedule, identical to the from-scratch reference (not tunable)
    max_steps = args.max_steps
    if is_main:
        print(f"[upscale:{args.op}] schedule {hp.version}: warmup={hp.warmup_steps} "
              f"cooldown_frac={hp.cooldown_frac} budget_steps={C.BIG_BUDGET_STEPS} "
              f"max_steps={max_steps}")
    sync = make_grad_sync(world)
    small_flops = load_checkpoint(args.ckpt, device)[1].get("flops", 0.0)
    flops_per_step_big = measured_flops_per_step(C.BIG_CONFIG, device, rank, world)
    small, _ = load_checkpoint(args.ckpt, device)
    res = run_upscale(
        small, op=args.op, target_config=C.BIG_CONFIG, hparams=hp,
        L1=args.l1, L2=args.l2, train_gen=new_train_gen(device, rank, world),
        val_batches=build_val_batches(device, rank, world), fit_steps=args.fit_steps,
        max_steps=max_steps, schedule_steps=C.BIG_BUDGET_STEPS, eval_every=args.eval_every,
        references_path=args.refs, ref_version=args.ref_version, small_flops=small_flops,
        records_path=(args.records if is_main else None),
        contributor=args.contributor, device=device, seed=3, log=is_main,
        amp_dtype=C.AMP_DTYPE, grad_sync=sync, flops_per_step=flops_per_step_big,
        world_size=world, compile_fn=maybe_compile,
    )
    if is_main:
        r = res.record
        c = r.get("c_amortized", "")
        print(f"[upscale:{args.op}] reached={r['reached']} steps_to_L2={r['steps_to_L2']} "
              f"flops={r['flops_warmstart']:.4g} c_amortized={c}")


PHASES = {"small": phase_small, "refs": phase_refs, "upscale": phase_upscale}


def main():
    ap = argparse.ArgumentParser(description="Run the upscaling benchmark at real scale")
    ap.add_argument("phase", choices=PHASES.keys())
    ap.add_argument("--l1", type=float, default=C.L1)
    ap.add_argument("--l2", type=float, default=C.L2)
    ap.add_argument("--op", choices=["stack_pad", "ligo"], default="stack_pad")
    ap.add_argument("--max-steps", type=int, default=40000,
                    help="loop cap only; the LR schedule horizon is the fixed per-model "
                    "config.SMALL_BUDGET_STEPS / BIG_BUDGET_STEPS. A run should reach its "
                    "target loss well before this cap.")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--fit-steps", type=int, default=500)
    ap.add_argument("--ref-version", default="ref1")
    ap.add_argument("--contributor", default="", help="optional handle logged to "
                    "records.csv; leave empty and fill the leaderboard in yourself")
    ap.add_argument("--ckpt", default="records/track_4_upscaling/checkpoints/small_L1.pt")
    ap.add_argument("--refs", default="records/track_4_upscaling/references.csv")
    ap.add_argument("--records", default="records/track_4_upscaling/records.csv")
    args = ap.parse_args()

    rank, world, _, device = setup_dist()
    ctx = (rank, world, device, rank == 0)
    try:
        PHASES[args.phase](args, ctx)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
