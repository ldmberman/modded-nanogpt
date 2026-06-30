"""Plot the small model's Chinchilla loss-vs-tokens fit and the chosen L1/L2.

Fits the per-size data term ``L(D) = E_floor + B * D**(-beta)`` to the
loss-vs-tokens ladder in ``floor_runs.csv`` (one model size, several token
budgets). ``E_floor`` is the small model's capacity floor: the lowest loss it
could reach with unlimited tokens. The figure shows the ladder, the fit, the
floor, and where L1 and L2 sit relative to it.

    python -m records.track_4_upscaling.utils.plot_floor
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import torch

from records.track_4_upscaling import config as C
from records.track_4_upscaling.model import GPT, count_params

HERE = Path(__file__).resolve().parent.parent
LADDER = HERE / "floor_runs.csv"
REFS_BIG = HERE / "refs_big.log"
OUT = HERE / "assets" / "floor_fit.png"
L1, L2 = 3.25, 3.18

TOK_PER_STEP = 262144
TOK_PER_PARAM = 20          # Chinchilla compute-optimal ratio
BETA_FIX = 0.43             # shared exponent from the small-model ladder fit
STEP_RE = re.compile(r"step\s+(\d+)\s*/\s*\d+\s+val_loss\s+([\d.]+)")


def load_ladder(path: Path) -> tuple[list[float], list[float]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(r["D"]) for r in rows], [float(r["L"]) for r in rows]


def parse_log(path: Path) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def fit_big(path: Path, band=(3.08, 3.30)):
    """Chinchilla fit of the from-scratch large run over its monotone descent."""
    steps, losses = parse_log(path)
    lo, hi = band
    D = [s * TOK_PER_STEP for s, l in zip(steps, losses) if lo <= l <= hi and s > 0]
    L = [l for s, l in zip(steps, losses) if lo <= l <= hi and s > 0]
    return fit_floor(D, L, beta_grid=[BETA_FIX])


def fit_floor(D: list[float], L: list[float], beta_grid=None):
    """Least-squares (E_floor, B) at each beta on a grid; keep the lowest RMSE."""
    Dt = torch.tensor(D, dtype=torch.float64)
    Lt = torch.tensor(L, dtype=torch.float64)
    beta_grid = beta_grid if beta_grid is not None else torch.linspace(0.25, 0.65, 81).tolist()
    best = None
    for b in beta_grid:
        X = torch.stack([torch.ones_like(Dt), Dt ** (-b)], dim=1)
        sol = torch.linalg.lstsq(X, Lt.unsqueeze(1)).solution.squeeze(1)
        rmse = float(((X @ sol - Lt) ** 2).mean().sqrt())
        E, B = float(sol[0]), float(sol[1])
        if best is None or rmse < best[-1]:
            best = (E, B, float(b), rmse)
    return best  # (E_floor, B, beta, rmse)


def tokens_for_loss(target: float, E: float, B: float, beta: float) -> float:
    """Invert L = E + B D**-beta for D (NaN if target <= floor)."""
    if target <= E:
        return float("nan")
    return ((target - E) / B) ** (-1.0 / beta)


def main():
    D, L = load_ladder(LADDER)
    E, B, beta, rmse = fit_floor(D, L)
    print(f"fit: E_floor={E:.3f} B={B:.4g} beta={beta:.3f} rmse={rmse:.4f}")
    d_l1, d_l2 = tokens_for_loss(L1, E, B, beta), tokens_for_loss(L2, E, B, beta)
    print(f"tokens@L1({L1})={d_l1:.3g}  tokens@L2({L2})={d_l2:.3g}")

    # Small-model compute-optimal loss at 20 tokens/param.
    n_small = count_params(GPT(C.SMALL_CONFIG), include_embedding=True)
    d_opt_s = TOK_PER_PARAM * n_small
    l_opt_s = E + B * d_opt_s ** (-beta)
    print(f"small compute-optimal: D={d_opt_s/1e9:.2f}B  loss={l_opt_s:.3f}")

    # Large-model Chinchilla fit from the from-scratch large run.
    Eb, Bb, betab, _ = fit_big(REFS_BIG)
    print(f"big fit: E_floor={Eb:.3f} B={Bb:.4g} beta={betab:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.logspace(torch.log10(torch.tensor(D[0] * 0.6)),
                          torch.log10(torch.tensor(max(d_l2 * 2, D[-1] * 50))), 400)
    curve = E + B * grid ** (-beta)
    big_curve = Eb + Bb * grid ** (-betab)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(grid / 1e9, curve, color="#1f77b4", lw=2,
            label=fr"small fit  $L = {E:.2f} + {B:.0f}\,D^{{-{beta:.2f}}}$")
    ax.plot(grid / 1e9, big_curve, color="#ff7f0e", lw=2,
            label=fr"large fit  $L = {Eb:.2f} + {Bb:.0f}\,D^{{-{betab:.2f}}}$")
    ax.scatter([d / 1e9 for d in D], L, color="#111", zorder=7, s=36)

    # Dots on the large curve at the same token marks as the small ladder.
    big_dots = [(t / 1e9, Eb + Bb * t ** (-betab)) for t in D[1:]]
    ax.scatter(*zip(*big_dots), color="#111", zorder=7, s=36)

    # The large curve from L1 onward is the ideal warm-start path; recolour and run it out.
    d_b_l1 = tokens_for_loss(L1, Eb, Bb, betab)
    seg = grid >= d_b_l1
    ax.plot(grid[seg] / 1e9, big_curve[seg], color="#9467bd", lw=2,
            zorder=4, label="ideal warm-start curve")

    # Reference losses as horizontal guides; labels live in the legend, not on the canvas.
    ax.axhline(E, ls="--", color="#888", lw=1.3, label=f"capacity floor  {E:.2f}")
    ax.axhline(l_opt_s, ls="--", color="#1f77b4", lw=1.3,
               label=f"small compute-optimal  {l_opt_s:.2f}")
    ax.axhline(L1, ls="--", color="#2ca02c", lw=1.3, label="L1 = 3.25 handed-out checkpoint")
    ax.axhline(L2, ls="--", color="#d62728", lw=1.3, label="L2 = 3.18 upscaling target")

    # Double arrow marking the L1 -> L2 gap and what it costs in wall-clock time.
    x_arrow = 9.0
    ax.annotate("", xy=(x_arrow, L2 - 0.008), xytext=(x_arrow, L1 + 0.008),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))
    ax.text(x_arrow * 1.15, (L1 + L2) / 2, "~30 min on 8xH100",
            color="#333", fontsize=9, ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.set_xscale("log")
    ax.set_ylim(2.9, 3.5)
    ax.set_xlabel(r"training tokens  ($\times 10^{9}$)")
    ax.set_ylabel("validation loss")
    ax.set_title("Loss vs tokens")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
