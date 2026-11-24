"""Quick summary visualizer for SectorPrognosis paradigm benchmark results."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "/tmp/sector_data/benchmark_results.json"
OUT_PNG = "/tmp/sector_data/paradigm_comparison.png"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    res = data["results"]

    names = list(res.keys())
    short_names = [n.split("_", 1)[1] for n in names]

    rmse = [res[n]["rmse"] for n in names]
    dir_acc = [res[n]["dir_acc"] * 100 for n in names]
    cs_ic = [res[n]["cs_ic"] for n in names]
    ls_sprd = [res[n]["ls_spread"] * 100 for n in names]

    colors = ["#7f8c8d", "#95a5a6", "#e74c3c", "#9b59b6", "#3498db", "#2ecc71", "#f39c12", "#1abc9c"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SectorPrognosis — Leak-Free Out-of-Sample Paradigm Comparison (Test: 2021–2024)", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], rmse, "RMSE (Lower is Better)", "RMSE", False),
        (axes[0, 1], dir_acc, "Directional Accuracy % (Higher is Better)", "% Correct Sign", True),
        (axes[1, 0], cs_ic, "Cross-Sectional Rotation IC (Higher is Better)", "Spearman IC", True),
        (axes[1, 1], ls_sprd, "Long-Short Top-2 vs Bottom-2 Spread % (Higher is Better)", "Annual Spread %", True),
    ]

    for ax, vals, title, ylabel, higher_better in panels:
        y_pos = np.arange(len(short_names))
        bars = ax.barh(y_pos, vals, color=colors[: len(short_names)], alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(ylabel, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        if not higher_better:
            ax.axvline(min(vals), color="black", linestyle="--", alpha=0.4)
        else:
            ax.axvline(0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Saved chart to {OUT_PNG}")


if __name__ == "__main__":
    main()
