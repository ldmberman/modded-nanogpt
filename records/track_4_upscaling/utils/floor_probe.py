"""Cooled floor probe for the small model.

Runs several *independent* cooled WSD runs at increasing token budgets (each
cools its LR to ~0 at its own endpoint, so its final val loss is a clean
frontier point), then fits the Chinchilla D-term ``L(D) = E_floor + B*D^-beta``
at fixed model size to extrapolate the floor and read the compute-optimal point.

Run on 8xH100:
    torchrun --standalone --nproc_per_node=8 -m records.track_4_upscaling.utils.floor_probe \
        --steps 3000,6000,10000,16000
"""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from ..model import set_seed, count_params, non_embedding_params
from ..training import train_model, append_csv
from .. import config as C
from ..run import (setup_dist, make_grad_sync, new_train_gen, build_val_batches,
                   measured_flops_per_step, build_model, maybe_compile)


CSV_FIELDS = ["run", "schedule", "step", "N", "D", "L"]


def fit_floor(points, beta_grid=None):
    """Fit L = E + B*D^-beta over (D, L) points; scan beta, LSQ for (E, B).
    points = [(D, L), ...]. Returns (E, B, beta, rmse)."""
    beta_grid = beta_grid or [round(x, 3) for x in torch.linspace(0.20, 0.60, 41).tolist()]
    D = torch.tensor([p[0] for p in points], dtype=torch.float64)
    L = torch.tensor([p[1] for p in points], dtype=torch.float64)
    best = None
    for beta in beta_grid:
        X = torch.stack([torch.ones_like(D), D ** (-beta)], dim=1)
        sol = torch.linalg.lstsq(X, L.unsqueeze(1)).solution.squeeze(1)
        rmse = float(((X @ sol - L) ** 2).mean().sqrt())
        E, B = float(sol[0]), float(sol[1])
        if best is None or rmse < best[3]:
            best = (E, B, beta, rmse)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="3000,6000,10000,16000",
                    help="comma-separated max_steps for the cooled ladder runs")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--csv", default="records/track_4_upscaling/floor_runs.csv")
    ap.add_argument("--token-param-ratio", type=float, default=20.0,
                    help="D_opt = ratio * N_nonembed for the compute-optimal L1 readout")
    args = ap.parse_args()

    budgets = [int(s) for s in args.steps.split(",") if s.strip()]
    rank, world, _, device = setup_dist()
    is_main = rank == 0
    sync = make_grad_sync(world)

    n_total = count_params(build_model(C.SMALL_CONFIG, torch.device("cpu"), seed=0), include_embedding=True)
    n_nonembed = non_embedding_params(build_model(C.SMALL_CONFIG, torch.device("cpu"), seed=0))
    flops_per_step = measured_flops_per_step(C.SMALL_CONFIG, device, rank, world)
    global_tokens = world * C.HPARAMS.batch_seqs * C.HPARAMS.seq_len
    if is_main:
        print(f"[floor] small N_total={n_total:,} N_nonembed={n_nonembed:,} "
              f"global_batch={global_tokens} tok/step  budgets={budgets}")

    val = build_val_batches(device, rank, world)
    points = []  # (D, L)
    for b in budgets:
        model = maybe_compile(build_model(C.SMALL_CONFIG, device, seed=1))
        res = train_model(model, new_train_gen(device, rank, world), val,
                          hparams=C.HPARAMS, max_steps=b, target_loss=None,
                          eval_every=args.eval_every, log=is_main,
                          amp_dtype=C.AMP_DTYPE, grad_sync=sync,
                          flops_per_step=flops_per_step)
        D = b * global_tokens
        L = res.final_loss
        points.append((D, L))
        if is_main:
            print(f"[floor] cooled run steps={b} D={D/1e9:.3f}B  val_loss={L:.4f}")
            append_csv(args.csv, {"run": "small_cooled_8xh100", "schedule": "wsd",
                                  "step": b, "N": n_total, "D": D, "L": L}, CSV_FIELDS)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main:
        E, B, beta, rmse = fit_floor(points)
        d_opt = args.token_param_ratio * n_nonembed
        l_opt = E + B * d_opt ** (-beta)
        print("\n[floor] ===== fit L(D) = E + B*D^-beta =====")
        print(f"[floor] E_floor={E:.4f}  B={B:.4g}  beta={beta:.3f}  rmse={rmse:.4f}")
        print(f"[floor] compute-optimal D_opt={d_opt/1e9:.3f}B (={args.token_param_ratio:g}x N_nonembed)"
              f"  =>  L_opt={l_opt:.4f}  (L1 candidate)")
        print(f"[floor] suggested L1 ~= {l_opt:.3f}  (well-trained small)")
        for delta in (0.05, 0.10, 0.15):
            print(f"[floor] suggested L2 = E_floor - {delta:.2f} = {E - delta:.3f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
