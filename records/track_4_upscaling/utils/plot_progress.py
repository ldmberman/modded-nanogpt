from __future__ import annotations

import csv
from pathlib import Path


def read_records(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def markdown_table(records: list[dict]) -> str:
    cols = ["op", "target_params", "L1", "L2", "steps_to_L2",
            "flops_warmstart", "c_amortized", "c_cold"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in records:
        cells = []
        for c in cols:
            fv = _to_float(r.get(c, ""))
            if c == "flops_warmstart" and fv is not None:
                cells.append(f"{fv:.3g}")
            elif c in ("c_amortized", "c_cold") and fv is not None:
                cells.append(f"{fv:.3f}")
            else:
                cells.append(str(r.get(c, "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(records: list[dict], out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(markdown_table(records) + "\n")
    made = {"table": str(out / "table.md"), "progress": None}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return made

    by_op: dict[str, list[float]] = {}
    for r in records:
        f = _to_float(r.get("flops_warmstart"))
        if f is not None:
            by_op.setdefault(r.get("op", "?"), []).append(f)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for op, ys in by_op.items():
        ax.plot(range(1, len(ys) + 1), ys, marker="o", label=op)
    ax.set_xlabel("submission #")
    ax.set_ylabel("warm-start FLOPs to L2")
    ax.set_yscale("log")
    ax.set_title("Cost-to-L2 progression")
    ax.legend()
    fig.tight_layout()
    path = out / "progress.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    made["progress"] = str(path)
    return made


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records/track_4_upscaling/records.csv")
    ap.add_argument("--out", default="records/track_4_upscaling/assets")
    args = ap.parse_args()
    records = read_records(args.records)
    made = plot(records, args.out)
    print(markdown_table(records))
    print("wrote:", made)


if __name__ == "__main__":
    main()
