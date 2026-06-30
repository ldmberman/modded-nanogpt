"""Shared FLOPs accounting for the upscaling runs.

Holds the measured per-step FLOP rates and the from-scratch reference costs that
place every run on a common cumulative-FLOPs axis, plus a Chinchilla fit over a
run's descent. The loss-vs-tokens fitting helpers live in ``plot_floor``.
"""
from __future__ import annotations

from pathlib import Path

from records.track_4_upscaling.utils.plot_floor import BETA_FIX, TOK_PER_STEP, fit_floor

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"

SMALL_FLOPS = 2.32904265302016e18       # checkpoints/small_L1.pt meta["flops"], at L1
SMALL_STEPS_TO_L1 = 10000               # small_L1.log horizon
SMALL_BPS = SMALL_FLOPS / SMALL_STEPS_TO_L1
SMALL_FPT = SMALL_BPS / TOK_PER_STEP

DENOM = 5.30725e18                       # from_scratch_large FLOPs to L2 (references.csv)
STEPS_TO_L2 = 13250
BIG_BPS = DENOM / STEPS_TO_L2            # big-model FLOPs per step, ~4.005e14
BIG_FPT = BIG_BPS / TOK_PER_STEP

FIT_FLOPS = {"stack_pad": 0.0, "ligo": 2.02e17}  # LiGO operator-fitting cost


def fit_run(steps, losses, band=(3.08, 3.30)):
    """Chinchilla fit L = E + B*tokens**-BETA_FIX over a run's monotone descent."""
    lo, hi = band
    D = [s * TOK_PER_STEP for s, l in zip(steps, losses) if lo <= l <= hi and s > 0]
    L = [l for s, l in zip(steps, losses) if lo <= l <= hi and s > 0]
    return fit_floor(D, L, beta_grid=[BETA_FIX])
