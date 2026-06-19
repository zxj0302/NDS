"""
For each real-world dataset in config.yaml, count positive/negative edges
and plot the weight distribution. Output saved to results/figure/.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"
OUTPUT_DIR = ROOT_DIR / "results" / "figure"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def read_dataset(path: Path):
    weights = []
    with open(path) as f:
        first_line = f.readline().split()
        n_vertices, n_edges = int(first_line[0]), int(first_line[1])
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                weights.append(float(parts[2]))
    return n_vertices, n_edges, np.array(weights)


def plot_all(datasets: list[tuple[str, Path]]):
    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes_flat = axes.flatten() if n > 1 else [axes]

    summary_rows = []

    for idx, (name, path) in enumerate(datasets):
        ax = axes_flat[idx]
        try:
            n_vertices, n_edges_header, weights = read_dataset(path)
        except Exception as e:
            ax.set_title(f"{name}\n(error reading)")
            ax.axis("off")
            continue

        pos = weights[weights > 0]
        neg = weights[weights < 0]
        n_pos, n_neg = len(pos), len(neg)
        total = len(weights)

        summary_rows.append((name, n_vertices, total, n_pos, n_neg))

        ax.hist(weights, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Weight", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)

        info = f"+: {n_pos}  −: {n_neg}"
        ax.text(
            0.97, 0.95, info,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Hide unused subplots
    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Real-World Dataset Weight Distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dataset_weight_distributions.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary table
    header = f"{'Dataset':<25} {'Vertices':>10} {'Edges':>10} {'Positive':>10} {'Negative':>10} {'Frac+':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, nv, total, n_pos, n_neg in summary_rows:
        frac = n_pos / total if total > 0 else float("nan")
        print(f"{name:<25} {nv:>10,} {total:>10,} {n_pos:>10,} {n_neg:>10,} {frac:>8.3f}")


def main():
    cfg = load_config()
    real_world_inputs = cfg.get("real-world", {}).get("input", [])

    datasets = []
    for entry in real_world_inputs:
        path_str = entry["path"] if isinstance(entry, dict) else entry
        path = ROOT_DIR / path_str.lstrip("./")
        name = path.stem
        if path.exists():
            datasets.append((name, path))
        else:
            print(f"Warning: {path} not found, skipping.")

    if not datasets:
        print("No datasets found.")
        return

    plot_all(datasets)


if __name__ == "__main__":
    main()
