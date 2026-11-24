"""Diagnostic visualization utilities for SectorPrognosis."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import SECTORS


def plot_iteration_progression(df_comp: pd.DataFrame, out_path: Path | str) -> None:
    """Bar charts comparing RMSE, OOS R² vs Train Mean, Cross-Sectional IC, and Top-2 Excess Return."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    models = df_comp["model"].tolist()
    x = np.arange(len(models))
    colors = ["#c44e52" if "Original" in m else ("#8172b2" if "Naive" in m else "#4c72b0") for m in models]
    if len(colors) > 0:
        colors[-1] = "#2ca02c"  # Highlight final ensemble

    specs = [
        ("rmse", "Out-of-Sample RMSE (Lower is Better)", axes[0, 0]),
        ("oos_r2_vs_train_mean", "OOS R² vs Historical Train Mean (Higher is Better)", axes[0, 1]),
        ("mean_cs_ic", "Cross-Sectional Spearman IC (Higher is Better)", axes[1, 0]),
        ("top2_excess_1y_ret", "Top-2 Sector Portfolio Excess 1Y Return vs EW", axes[1, 1]),
    ]
    for col, title, ax in specs:
        vals = df_comp[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            offset = 0.01 * (np.nanmax(np.abs(vals)) + 1e-4)
            va = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, v + (offset if v >= 0 else -offset), f"{v:.3f}",
                    ha="center", va=va, fontsize=8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sector_predictions_grid(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path | str,
) -> None:
    """2x3 grid plotting Predicted vs Actual 1-Year Forward Returns across all 6 sectors."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    for j, s in enumerate(SECTORS):
        ax = axes[j]
        ax.plot(dates, y_true[:, j] * 100, label="Actual 1Y Return (%)", color="#1f77b4", linewidth=1.4)
        ax.plot(dates, y_pred[:, j] * 100, label="Predicted 1Y Return (%)", color="#ff7f0e", linewidth=1.4, alpha=0.9)
        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(s, fontsize=11, fontweight="bold")
        ax.set_ylabel("1Y Return (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
