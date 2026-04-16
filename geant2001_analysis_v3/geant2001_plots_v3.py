#!/usr/bin/env python3
"""Vẽ bộ hình tổng quát cho Geant2001."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geant2001_common import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    build_convergence_dataframe,
    build_inference_dataframe,
    build_timing_breakdown_dataframe,
    build_tm_stats_dataframe,
    configure_plot_style,
    require_paths,
    save_figure,
)

def main() -> None:
    configure_plot_style()
    _, tm_dir, pckl_dir = require_paths("graph", "tm_dir", "pckl_dir")
    inference_df = build_inference_dataframe(pckl_dir=pckl_dir, tm_ids=range(50))
    convergence_df = build_convergence_dataframe(pckl_dir=pckl_dir, tm_ids=range(50))
    timing_df = build_timing_breakdown_dataframe(inference_df, convergence_df)
    tm_stats_df = build_tm_stats_dataframe(tm_dir=tm_dir, tm_ids=range(50))
    merged_df = inference_df.merge(tm_stats_df, on="tm_id", how="left").merge(timing_df, on="tm_id", how="left")

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    x = inference_df["tm_id"].to_numpy()
    ax.scatter(x, inference_df["mlu_init"], color=METHOD_COLORS["ospf"], marker=METHOD_MARKERS["ospf"], s=54, label=METHOD_LABELS["ospf"])
    ax.scatter(x, inference_df["mlu_drl"], color=METHOD_COLORS["drl"], marker=METHOD_MARKERS["drl"], s=42, label=METHOD_LABELS["drl"])
    ax.scatter(x, inference_df["mlu_ls"], color=METHOD_COLORS["ls"], marker=METHOD_MARKERS["ls"], s=42, label=METHOD_LABELS["ls"])
    ax.scatter(x, inference_df["mlu_enero"], color=METHOD_COLORS["enero"], marker=METHOD_MARKERS["enero"], s=58, label=METHOD_LABELS["enero"])
    for row in inference_df.itertuples():
        ax.plot([row.tm_id, row.tm_id], [row.mlu_init, row.mlu_enero], color="0.85", linewidth=1.0, zorder=0)
    ax.set_xlabel("Traffic matrix ID")
    ax.set_ylabel("MLU")
    ax.set_title("MLU của 4 phương pháp trên 50 traffic matrix")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    save_figure(fig, "fig1_mlu_before_after")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    axes[0].scatter(merged_df["mlu_init"], merged_df["improve_enero"], s=82, color=METHOD_COLORS["enero"], alpha=0.85)
    axes[0].scatter(merged_df["mlu_init"], merged_df["improve_ls"], s=52, color=METHOD_COLORS["ls"], alpha=0.6)
    axes[0].set_xlabel("MLU ban đầu")
    axes[0].set_ylabel("Cải thiện so với OSPF (%)")
    axes[0].set_title("Bài toán càng khó thì Enero càng có dư địa cải thiện")
    axes[0].grid(True, alpha=0.25)

    width = 0.26
    axes[1].bar(np.arange(len(merged_df)), merged_df["cost_drl"], width=width, color=METHOD_COLORS["drl"], label="DRL")
    axes[1].bar(np.arange(len(merged_df)) + width, merged_df["hc_only_time_s"], width=width, color=METHOD_COLORS["ls"], label="HC-only")
    axes[1].bar(np.arange(len(merged_df)) + 2 * width, merged_df["cost_enero"], width=width, color=METHOD_COLORS["enero"], label="Enero")
    axes[1].set_xlabel("Traffic matrix ID")
    axes[1].set_ylabel("Thời gian (s)")
    axes[1].set_title("Breakdown thời gian theo TM")
    axes[1].grid(True, alpha=0.25, axis="y")
    axes[1].legend()
    save_figure(fig, "fig2_improvement_and_cost")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for key in ("ospf", "drl", "ls", "enero"):
        column = {"ospf": "mlu_init", "drl": "mlu_drl", "ls": "mlu_ls", "enero": "mlu_enero"}[key]
        values = np.sort(inference_df[column].to_numpy(dtype=float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, color=METHOD_COLORS[key], linewidth=2.0, marker=None, label=METHOD_LABELS[key])
    ax.set_xlabel("MLU")
    ax.set_ylabel("CDF")
    ax.set_title("CDF của MLU")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig3_cdf_mlu")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    scatter = ax.scatter(
        merged_df["total_bw"],
        merged_df["cost_enero"],
        c=merged_df["improve_enero"],
        s=90,
        cmap="RdYlGn",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_xlabel("Tổng băng thông TM")
    ax.set_ylabel("Thời gian Enero (s)")
    ax.set_title("Tải tổng hợp và chi phí tối ưu")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Cải thiện Enero (%)")
    save_figure(fig, "fig4_total_bw_vs_enero_cost")
    plt.close(fig)

if __name__ == "__main__":
    main()
