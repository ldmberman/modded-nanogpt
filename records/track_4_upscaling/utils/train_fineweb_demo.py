# Tiny end-to-end check on real FineWeb tokens to run on CPU.
#
#   python data/cached_fineweb10B.py 1   # val shard + 1 train shard (~400MB)
#
# Runs as a 2-rank gloo group by default (self-spawned, no torchrun needed), which
# exercises grad sync, val all-reduce, Muon all-gather, and LiGO fit averaging:
#   python -m records.track_4_upscaling.utils.train_fineweb_demo
# DEMO_WORLD=1 for a single process; or launch under torchrun to pick the world size:
#   DEMO_WORLD=1 python -m records.track_4_upscaling.utils.train_fineweb_demo
#   torchrun --standalone --nproc_per_node=2 -m records.track_4_upscaling.utils.train_fineweb_demo

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import replace

import torch
import torch.distributed as dist

from ..model import (
    GPT, GPTConfig, init_params, non_embedding_params, set_seed, build_optimizers,
    measure_step_flops, GPT2_VOCAB_SIZE, fineweb_data_generator, fineweb_val_batches,
)
from ..training import (
    Hyperparameters, train_model, save_checkpoint, load_checkpoint, write_reference,
)
from ..train_upscaled_gpt import run_upscale
from ..run import make_grad_sync
from . import plot_progress as pp

DEVICE = torch.device("cpu")

TRAIN_GLOB = "data/fineweb10B/fineweb_train_*.bin"
VAL_GLOB = "data/fineweb10B/fineweb_val_*.bin"


HPARAMS = Hyperparameters(
    version="H1-fineweb",
    batch_seqs=8,
    seq_len=128,
    embed_lr=0.1,
    head_lr=0.02,
    scalar_lr=0.02,
    muon_lr=0.02,
    muon_wd=0.05,
    cooldown_frac=0.4,
    warmup_steps=20,
)


def _check_data():
    from pathlib import Path
    if not sorted(Path.cwd().glob(TRAIN_GLOB)) or not sorted(Path.cwd().glob(VAL_GLOB)):
        raise SystemExit(
            "FineWeb shards not found. From the repo root run:\n"
            "    python data/cached_fineweb10B.py 1\n"
            "to fetch the val shard + one train shard (~400MB)."
        )


def _run(rank: int, world: int):
    is_main = rank == 0
    sync = make_grad_sync(world)

    def pr(*a):
        if is_main:
            print(*a)

    def barrier():
        if world > 1:
            dist.barrier()

    _check_data()
    set_seed(0)
    t_start = time.time()

    val_batches = fineweb_val_batches(VAL_GLOB, batch_seqs=16, seq_len=HPARAMS.seq_len,
                                      device=DEVICE, total_tokens=8 * 16 * HPARAMS.seq_len,
                                      rank=rank, world_size=world)

    def new_gen():
        # fresh per-rank stream over the train shard(s); the loader wraps around if exhausted
        return fineweb_data_generator(TRAIN_GLOB, HPARAMS.batch_seqs, HPARAMS.seq_len, DEVICE,
                                      rank=rank, world_size=world)

    def measure_flops_per_step(config: GPTConfig) -> float:
        # measured FLOPs of one (global) training step: one rank's step x world_size,
        # matching the convention used by run.py and ligo's fit-step accounting
        m = GPT(config).to(DEVICE)
        init_params(m)
        opts = build_optimizers(m, embed_lr=HPARAMS.embed_lr, head_lr=HPARAMS.head_lr,
                                scalar_lr=HPARAMS.scalar_lr, muon_lr=HPARAMS.muon_lr,
                                muon_wd=HPARAMS.muon_wd)
        return float(measure_step_flops(m, next(new_gen()), optimizers=opts)) * world

    rand_loss = torch.log(torch.tensor(float(GPT2_VOCAB_SIZE))).item()
    pr(f"[dist] world_size={world} backend={'gloo' if world > 1 else 'none'}")
    pr(f"[data] FineWeb GPT-2 tokens, vocab={GPT2_VOCAB_SIZE}, random-guess loss ~= {rand_loss:.3f}")

    # shared artifact dir: rank 0 picks it, everyone uses the same path (single node)
    tmp = tempfile.mkdtemp(prefix="track4_fineweb_") if is_main else None
    if world > 1:
        obj = [tmp]
        dist.broadcast_object_list(obj, src=0)
        tmp = obj[0]
    ckpt_path = os.path.join(tmp, "small_L1.pt")
    refs_path = os.path.join(tmp, "references.csv")
    records_path = os.path.join(tmp, "records.csv")

    # ----- small model ------------------------------------------------------
    small_cfg = GPTConfig(GPT2_VOCAB_SIZE, num_layers=2, model_dim=128, head_dim=16)
    flops_per_step_small = measure_flops_per_step(small_cfg)
    pr(f"\n[1] probe small ({small_cfg.num_layers}L x {small_cfg.model_dim}d, "
       f"{non_embedding_params(GPT(small_cfg))/1e3:.0f}K non-embed params)")
    set_seed(1)
    probe = GPT(small_cfg).to(DEVICE)
    init_params(probe)
    probe_res = train_model(probe, new_gen(), val_batches, hparams=HPARAMS, max_steps=120,
                            eval_every=10, log=is_main, flops_per_step=flops_per_step_small,
                            grad_sync=sync)
    Lstart, Lend = probe_res.loss_history[0][1], probe_res.final_loss
    rng = Lstart - Lend
    L1 = Lend + 0.22 * rng   # warm-start checkpoint loss (small meaningfully trained)
    L2 = Lend + 0.06 * rng   # target loss the upscaled model must reach
    pr(f"    probe Lstart={Lstart:.3f} Lend={Lend:.3f} -> L1={L1:.3f} L2={L2:.3f}")
    assert L1 > L2 > Lend

    pr("\n[2] train small to L1 and checkpoint")
    set_seed(1)
    small = GPT(small_cfg).to(DEVICE)
    init_params(small)
    small_res = train_model(small, new_gen(), val_batches, hparams=HPARAMS, max_steps=200,
                           target_loss=L1, eval_every=10, flops_per_step=flops_per_step_small,
                           grad_sync=sync)
    small_flops = small_res.flops
    pr(f"    reached L1={small_res.reached_target} at step {small_res.steps} (loss {small_res.final_loss:.3f})")

    if is_main:
        save_checkpoint(ckpt_path, small, meta={"L1": L1, "flops": small_flops})
    barrier()
    small, _ = load_checkpoint(ckpt_path, DEVICE)
    pr(f"    saved+reloaded -> {ckpt_path}")

    # ----- reference (from-scratch-large to L2) ---------------------------
    pr("\n[3] reference (from-scratch-large to L2)")
    # wider+deeper big model, mirroring the real small->big (both width and depth
    # grow), so the smoke test exercises the same stack_pad op as the leaderboard.
    big_cfg = replace(small_cfg, num_layers=3, model_dim=160)
    flops_per_step_big = measure_flops_per_step(big_cfg)

    set_seed(2)
    fs = GPT(big_cfg).to(DEVICE)
    init_params(fs)
    r_fs = train_model(fs, new_gen(), val_batches, hparams=HPARAMS, max_steps=300,
                       target_loss=L2, eval_every=10, flops_per_step=flops_per_step_big,
                       grad_sync=sync)
    if is_main:
        write_reference(refs_path, ref_version="ref1", kind="from_scratch_large",
                        model=fs, L1=L1, L2=L2, result=r_fs, seed=2,
                        hparams_version=HPARAMS.version)
    pr(f"    from_scratch_large reached={r_fs.reached_target} steps={r_fs.steps} flops={r_fs.flops:.3g}")
    barrier()  # references.csv must be visible to all ranks before upscale reads it

    # ----- warm-start: grow small -> upscaled (stack_pad and LiGO), resume to L2 --
    pr("\n[4] warm-start, resume to L2 -> records.csv")

    small_sp, _ = load_checkpoint(ckpt_path, DEVICE)
    res_sp = run_upscale(small_sp, op="stack_pad", target_config=big_cfg, hparams=HPARAMS, L1=L1, L2=L2,
                         train_gen=new_gen(), val_batches=val_batches,
                         max_steps=300, eval_every=10, references_path=refs_path, ref_version="ref1",
                         small_flops=small_flops, records_path=(records_path if is_main else None),
                         contributor="fineweb_demo", seed=3, flops_per_step=flops_per_step_big,
                         grad_sync=sync)
    rd = res_sp.record
    pr(f"    stack_pad  : reached={rd['reached']} steps_to_L2={rd['steps_to_L2']} "
       f"c_amortized={float(rd['c_amortized']):.3f}")

    small_ligo, _ = load_checkpoint(ckpt_path, DEVICE)
    res_ligo = run_upscale(small_ligo, op="ligo", target_config=big_cfg, hparams=HPARAMS, L1=L1, L2=L2,
                           train_gen=new_gen(), val_batches=val_batches, fit_steps=40,
                           max_steps=300, eval_every=10, references_path=refs_path, ref_version="ref1",
                           small_flops=small_flops, records_path=(records_path if is_main else None),
                           contributor="fineweb_demo", seed=3, flops_per_step=flops_per_step_big,
                           grad_sync=sync)
    rl = res_ligo.record
    pr(f"    LiGO       : reached={rl['reached']} steps_to_L2={rl['steps_to_L2']} "
       f"c_amortized={float(rl['c_amortized']):.3f}")

    if is_main:
        made = pp.plot(pp.read_records(records_path), os.path.join(tmp, "assets"))
        print("\n" + pp.markdown_table(pp.read_records(records_path)))
        print("\n[note] The toy scale here is not meant to set a record: with a tiny model,")
        print("       ~1e3 steps and 100M tokens, the grown model has no budget to amortize,")
        print("       so c >= 1 is expected. The methods are designed to pay off at real")
        print("       scale with substantial continued pretraining.")
        print(f"\nDONE on real FineWeb tokens in {time.time()-t_start:.0f}s")
        print(f"(artifacts in {tmp}, plots: {made})")

    barrier()  # process-group teardown is handled by the caller


def _worker(rank: int, world: int):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // world))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        _run(rank, world)
    finally:
        dist.destroy_process_group()


def main():
    # Launched under torchrun (e.g. on the H100 box): just join the existing group.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="gloo")
        try:
            _run(dist.get_rank(), dist.get_world_size())
        finally:
            dist.destroy_process_group()
        return

    # Plain `python -m ...`: self-spawn a 2-rank gloo group (DEMO_WORLD=1 for single).
    world = int(os.environ.get("DEMO_WORLD", "2"))
    if world <= 1:
        _run(0, 1)
        return
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(world,), nprocs=world, join=True)


if __name__ == "__main__":
    main()
