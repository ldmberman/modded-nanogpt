"""Zoomed loss-vs-FLOPs view of the warm-start runs around the L1 -> L2 band.

Companion to ``plot_floor``: same style, but the x-axis is cumulative training
FLOPs and the two baseline warm-starts are shown against the from-scratch large
run and the ideal warm-start. Warm-starts start at the checkpoint cost, so on a
cumulative axis they sit to the right of the from-scratch large run; the ``c``
metric compares only the marginal big-model FLOPs.

    python -m records.track_4_upscaling.utils.plot_flops
"""
from __future__ import annotations

from pathlib import Path

import torch

from records.track_4_upscaling.utils.plot_floor import (
    BETA_FIX,
    L1,
    L2,
    LADDER,
    fit_floor,
    load_ladder,
    parse_log as parse,
    tokens_for_loss,
)
from records.track_4_upscaling.utils.flops_accounting import (
    BIG_BPS,
    BIG_FPT,
    FIT_FLOPS,
    RESULTS,
    SMALL_BPS,
    SMALL_FLOPS,
    SMALL_FPT,
    fit_run,
)

HERE = Path(__file__).resolve().parent.parent
REFS_BIG = HERE / "refs_big.log"
SMALL_L1 = HERE / "small_L1.log"
OUT = HERE / "assets" / "flops_fit.png"

XLO, XHI = 1.0, 8.5          # cumulative-FLOPs window (x10^18)
YLO, YHI = 3.15, 3.40


def plot_line(ax, path: Path, base_flops: float, bps: float, color: str, label: str):
    """Trajectory on the cumulative-FLOPs axis, clipped to the zoom window."""
    steps, losses = parse(path)
    pts = [((base_flops + s * bps) / 1e18, l)
           for s, l in zip(steps, losses) if YLO - 0.04 <= l <= YHI + 0.25]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.8, label=label)


def main():
    bsteps, blosses = parse(REFS_BIG)
    Eb, Bb, _, _ = fit_run(bsteps, blosses)
    Es, Bs, betas, _ = fit_floor(*load_ladder(LADDER))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    plot_line(ax, SMALL_L1, 0.0, SMALL_BPS, "#111111", "small from scratch")

    # Continue the small run past L1 with its Chinchilla fit: it keeps descending,
    # but shallowly toward its own floor, far slower per FLOP than the big-model curves.
    dd = torch.linspace(tokens_for_loss(L1, Es, Bs, betas), XHI * 1e18 / SMALL_FPT, 120)
    ax.plot(dd * SMALL_FPT / 1e18, Es + Bs * dd ** (-betas),
            color="#111111", lw=1.5, ls=(0, (5, 3)), label="small from scratch, projected")

    plot_line(ax, REFS_BIG, 0.0, BIG_BPS, "#ff7f0e", "large from scratch, c = 1")
    plot_line(ax, RESULTS / "upscale_stack_pad.log", SMALL_FLOPS + FIT_FLOPS["stack_pad"],
              BIG_BPS, "#1f77b4", "stack_pad warm-start, c = 0.89")
    plot_line(ax, RESULTS / "upscale_ligo_lr0.5.log", SMALL_FLOPS + FIT_FLOPS["ligo"],
              BIG_BPS, "#d62728", "LiGO warm-start, c = 0.99")

    # Ideal warm-start: free to L1 via the checkpoint, then big-model marginal FLOPs to L2.
    def flops_big(loss):
        return ((loss - Eb) / Bb) ** (-1.0 / BETA_FIX) * BIG_FPT

    f_l1 = flops_big(L1)
    lg = torch.linspace(L1, L2, 80)
    x_ideal = (SMALL_FLOPS + (flops_big(lg) - f_l1)) / 1e18
    ax.plot(x_ideal, lg, color="#9467bd", lw=2, label="ideal warm-start")

    ax.axhline(L1, ls="--", color="#888", lw=1.3, label="L1 = 3.25 handed-out checkpoint")
    ax.axhline(L2, ls="--", color="#333", lw=1.3, label="L2 = 3.18 upscaling target")
    ax.axvline(SMALL_FLOPS / 1e18, ls=":", color="#9467bd", lw=1.1)
    ax.text(SMALL_FLOPS / 1e18, 3.155, "checkpoint at L1", color="#9467bd",
            fontsize=9, rotation=90, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(r"cumulative training FLOPs  ($\times 10^{18}$)")
    ax.set_ylabel("validation loss")
    ax.set_title("Loss vs cumulative FLOPs")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
