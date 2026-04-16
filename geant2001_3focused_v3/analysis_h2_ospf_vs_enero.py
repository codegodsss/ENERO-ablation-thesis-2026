#!/usr/bin/env python3
"""So sánh trọng tâm giữa OSPF, DRL, LS và Enero trên Geant2001."""
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
    build_convergence_dataframe,
    build_inference_dataframe,
    build_timing_breakdown_dataframe,
    configure_plot_style,
    safe_ci95,
    save_figure,
)

def main() -> None:
    configure_plot_style()
    inference_df = build_inference_dataframe(tm_ids=range(50))
    convergence_df = build_convergence_dataframe(tm_ids=range(50))
    timing_df = build_timing_breakdown_dataframe(inference_df, convergence_df)
    df = inference_df.merge(timing_df, on="tm_id", how="left", suffixes=("", "_timing"))

    mi = df["mlu_init"].to_numpy(dtype=float)
    md = df["mlu_drl"].to_numpy(dtype=float)
    ml = df["mlu_ls"].to_numpy(dtype=float)
    me = df["mlu_enero"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    data = [mi, md, ml, me]
    colors = [METHOD_COLORS["ospf"], METHOD_COLORS["drl"], METHOD_COLORS["ls"], METHOD_COLORS["enero"]]
    labels = ["OSPF/SP", "DRL only", "LS (HC)", "Enero"]
    violin = ax.violinplot(data, positions=[1, 2, 3, 4], showmedians=True, widths=0.58)
    for patch, color in zip(violin["bodies"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        violin[key].set_color("black")
    ax.boxplot(data, positions=[1, 2, 3, 4], widths=0.18, patch_artist=True, medianprops={"color": "black"})
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels)
    ax.set_ylabel("MLU")
    ax.set_title("Phân bố MLU của 4 phương pháp")
    ax.grid(True, alpha=0.25, axis="y")
    save_figure(fig, "fig_h2a_violin_4methods")
    plt.close(fig)

    order = np.argsort(df["improve_enero"].to_numpy(dtype=float))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    x = np.arange(len(order))
    axes[0].plot(x, df["improve_enero"].to_numpy()[order], color=METHOD_COLORS["enero"], marker="D", markersize=4, linewidth=1.6, label="Enero")
    axes[0].plot(x, df["improve_ls"].to_numpy()[order], color=METHOD_COLORS["ls"], marker="s", markersize=3.8, linewidth=1.3, label="LS")
    axes[0].plot(x, df["improve_drl"].to_numpy()[order], color=METHOD_COLORS["drl"], marker="o", markersize=3.5, linewidth=1.1, label="DRL")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_xlabel("TM được sắp theo gain của Enero")
    axes[0].set_ylabel("Cải thiện so với OSPF (%)")
    axes[0].set_title("Profile gain theo từng TM")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    mean_values = [df["improve_drl"].mean(), df["improve_ls"].mean(), df["improve_enero"].mean()]
    ci_values = [safe_ci95(df["improve_drl"]), safe_ci95(df["improve_ls"]), safe_ci95(df["improve_enero"])]
    bars = axes[1].bar(
        ["DRL", "LS", "Enero"],
        mean_values,
        color=[METHOD_COLORS["drl"], METHOD_COLORS["ls"], METHOD_COLORS["enero"]],
        yerr=ci_values,
        capsize=7,
    )
    for bar, mean_value in zip(bars, mean_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, mean_value + 0.35, f"{mean_value:.1f}%", ha="center", fontweight="bold")
    axes[1].set_ylabel("Mean improvement (%) ± 95% CI")
    axes[1].set_title("Gain trung bình so với OSPF")
    axes[1].grid(True, alpha=0.25, axis="y")
    save_figure(fig, "fig_h2b_improvement_ci")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for method, values in {
        "ospf": mi,
        "drl": md,
        "ls": ml,
        "enero": me,
    }.items():
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.plot(sorted_values, cdf, linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel("MLU")
    ax.set_ylabel("CDF")
    ax.set_title("CDF của MLU")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig_h2c_cdf_mlu")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    width = 0.32
    x = np.arange(len(df))
    ax.bar(x - width, df["best_drl_hit_time_s"], width=width, color=METHOD_COLORS["drl"], label="Thời điểm DRL đạt best")
    ax.bar(x, df["drl_tail_after_best_s"], width=width, color="#b07aa1", label="Đuôi DRL sau khi đã đạt best")
    ax.bar(x + width, df["hc_only_time_s"], width=width, color=METHOD_COLORS["ls"], label="HC-only")
    ax.set_xlabel("Traffic matrix ID")
    ax.set_ylabel("Thời gian (s)")
    ax.set_title("Breakdown thời gian của pipeline Enero")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    save_figure(fig, "fig_h2d_time_quality_tradeoff")
    plt.close(fig)

if __name__ == "__main__":
    main()
