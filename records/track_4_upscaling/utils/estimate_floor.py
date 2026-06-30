from __future__ import annotations

import csv
from dataclasses import dataclass

import torch


@dataclass
class ScalingFit:
    E: float
    A: float
    B: float
    a: float
    b: float
    rmse: float

    def loss(self, N: float, D: float) -> float:
        return self.E + self.A * N ** (-self.a) + self.B * D ** (-self.b)

    def floor(self, N: float) -> float:
        return self.E + self.A * N ** (-self.a)


def fit_fixed_exponents(points: list[tuple[float, float, float]], a: float, b: float) -> ScalingFit:
    """Linear LSQ for (E, A, B) at fixed (a, b). points = [(N, D, L), ...]."""
    N = torch.tensor([p[0] for p in points], dtype=torch.float64)
    D = torch.tensor([p[1] for p in points], dtype=torch.float64)
    L = torch.tensor([p[2] for p in points], dtype=torch.float64)
    X = torch.stack([torch.ones_like(N), N ** (-a), D ** (-b)], dim=1)  # [n, 3]
    sol = torch.linalg.lstsq(X, L.unsqueeze(1)).solution.squeeze(1)
    E, A, B = (float(v) for v in sol)
    pred = X @ sol
    rmse = float(((pred - L) ** 2).mean().sqrt())
    return ScalingFit(E=E, A=A, B=B, a=a, b=b, rmse=rmse)


def fit_scan(points: list[tuple[float, float, float]],
             a_grid=None, b_grid=None) -> ScalingFit:
    """Scan (a, b) and keep the lowest-RMSE fit. Use only with >= 3 distinct N."""
    a_grid = a_grid if a_grid is not None else [round(x, 3) for x in torch.linspace(0.2, 0.7, 11).tolist()]
    b_grid = b_grid if b_grid is not None else [round(x, 3) for x in torch.linspace(0.2, 0.6, 9).tolist()]
    best = None
    for a in a_grid:
        for b in b_grid:
            f = fit_fixed_exponents(points, a, b)
            if best is None or f.rmse < best.rmse:
                best = f
    return best


def load_ladder(path: str) -> list[tuple[float, float, float]]:
    with open(path, newline="") as f:
        return [(float(r["N"]), float(r["D"]), float(r["L"])) for r in csv.DictReader(f)]


# The data points from the track 1 runs.
IN_REPO_ANCHORS = [
    (1.62e8, 2.67e9, 3.275),
    (1.63e9, 7.50e9, 2.7672),
    (1.63e9, 1.00e10, 2.7297),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", help="CSV with columns N,D,L")
    ap.add_argument("--a", type=float, default=0.49)
    ap.add_argument("--b", type=float, default=0.38)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--floor-at", type=float, nargs="*", default=[])
    args = ap.parse_args()

    points = load_ladder(args.ladder) if args.ladder else IN_REPO_ANCHORS
    fit = fit_scan(points) if args.scan else fit_fixed_exponents(points, args.a, args.b)
    print(f"fit: E={fit.E:.4f} A={fit.A:.4g} B={fit.B:.4g} a={fit.a} b={fit.b} rmse={fit.rmse:.4g}")
    for N in args.floor_at:
        print(f"  L_inf(N={N:.3g}) = {fit.floor(N):.4f}")


if __name__ == "__main__":
    main()
