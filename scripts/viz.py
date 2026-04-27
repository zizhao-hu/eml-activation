"""Plot loss curves and benchmark bars."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"relu": "#888888", "gelu": "#1f77b4", "swiglu": "#2ca02c", "emlglu": "#d62728"}


def load_loss(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps, train, val = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def plot_loss_curves(runs_dir: Path, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind in ("relu", "gelu", "swiglu", "emlglu"):
        run = runs_dir / kind / "loss.csv"
        if not run.exists():
            print(f"skip missing {run}")
            continue
        steps, train, val = load_loss(run)
        axes[0].plot(steps, train, label=kind, color=COLORS[kind], linewidth=1.5)
        axes[1].plot(steps, val, label=kind, color=COLORS[kind], linewidth=1.5)
    for ax, title in zip(axes, ["train loss", "val loss"]):
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy (nats)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def plot_bench(bench_csv: Path, out_path: Path):
    rows = list(csv.DictReader(bench_csv.open()))
    Ts = sorted({int(r["T"]) for r in rows})
    kinds = ["relu", "gelu", "swiglu", "emlglu"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bar_w = 0.2
    for i, kind in enumerate(kinds):
        fwd = [float(next(r for r in rows if r["ffn"] == kind and int(r["T"]) == T)["fwd_ms"]) for T in Ts]
        fb = [float(next(r for r in rows if r["ffn"] == kind and int(r["T"]) == T)["fb_ms"]) for T in Ts]
        x = [j + (i - 1.5) * bar_w for j in range(len(Ts))]
        axes[0].bar(x, fwd, bar_w, label=kind, color=COLORS[kind])
        axes[1].bar(x, fb, bar_w, label=kind, color=COLORS[kind])
    for ax, title in zip(axes, ["forward ms", "forward+backward ms"]):
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([f"T={T}" for T in Ts])
        ax.set_ylabel("ms / step (median)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="runs/figs")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(runs_dir, out_dir / "loss_curves.png")
    bench = runs_dir / "bench.csv"
    if bench.exists():
        plot_bench(bench, out_dir / "bench.png")


if __name__ == "__main__":
    main()
