"""Pick the per-model WSD schedule budgets using Chinchilla.

Each model trains under one fixed LR schedule sized to a realistic budget for that
model - enough tokens to land halfway in loss between its compute-optimal point
(~20 tokens/param) and its capacity floor, i.e. mild overtraining. The small and
big models have different optima and floors, so they get different budgets.

    python -m records.track_4_upscaling.utils.chinchilla_budget
"""
from __future__ import annotations

from pathlib import Path

from records.track_4_upscaling import config as C
from records.track_4_upscaling.model import GPT, count_params
from records.track_4_upscaling.utils.plot_floor import fit_floor, load_ladder, parse_log as parse
from records.track_4_upscaling.utils.flops_accounting import fit_run

HERE = Path(__file__).resolve().parent.parent
WORLD = 8
TOK_PER_STEP = C.HPARAMS.batch_seqs * C.HPARAMS.seq_len * WORLD
TOK_PER_PARAM = 20  # Chinchilla compute-optimal ratio


def budget_for(name: str, n_total: int, E: float, B: float, beta: float) -> int:
    """Steps for a schedule whose budget sits halfway in loss between the
    compute-optimal point and the floor of a model with `n_total` params."""
    d_opt = TOK_PER_PARAM * n_total
    l_opt = E + B * d_opt ** (-beta)
    l_budget = E + 0.5 * (l_opt - E)
    d_budget = ((l_budget - E) / B) ** (-1.0 / beta)
    steps = round(d_budget / TOK_PER_STEP)
    onset = round((1 - C.HPARAMS.cooldown_frac) * steps)
    print(f"[{name}] params={n_total/1e6:.1f}M  floor={E:.3f}  beta={beta:.2f}")
    print(f"    compute-optimal: D_opt={d_opt/1e9:.2f}B tokens  loss~{l_opt:.3f}")
    print(f"    budget (halfway to floor): loss~{l_budget:.3f}  D={d_budget/1e9:.2f}B tokens")
    print(f"    -> {steps} steps  (cooldown starts ~{onset}; target reached well before)")
    return steps


def main():
    n_small = count_params(GPT(C.SMALL_CONFIG), include_embedding=True)
    Es, Bs, betas, _ = fit_floor(*load_ladder(HERE / "floor_runs.csv"))
    small_steps = budget_for("small", n_small, Es, Bs, betas)

    n_big = count_params(GPT(C.BIG_CONFIG), include_embedding=True)
    steps_b, losses_b = parse(HERE / "refs_big.log")
    Eb, Bb, betab, _ = fit_run(steps_b, losses_b)
    big_steps = budget_for("big", n_big, Eb, Bb, betab)

    print(f"\nSMALL_BUDGET_STEPS = {small_steps}")
    print(f"BIG_BUDGET_STEPS   = {big_steps}")


if __name__ == "__main__":
    main()
