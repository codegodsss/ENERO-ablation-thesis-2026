#!/usr/bin/env python3
"""Phân tích failure mode giữa pha DRL và pha HC trên Geant2001."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geant2001_common import (
    METHOD_COLORS,
    OUTPUT_DIR,
    build_convergence_dataframe,
    build_inference_dataframe,
    build_timing_breakdown_dataframe,
    configure_plot_style,
    save_figure,
)

QMAP = {
    (True, True): ("Q1", "Q1: Both good", "#2a9d8f"),
    (False, True): ("Q2", "Q2: HC rescues DRL", "#4e79a7"),
    (True, False): ("Q3", "Q3: DRL sufficient", "#f4a261"),
    (False, False): ("Q4", "Q4: Both struggle", "#d1495b"),
}

def assign_quadrant(row: pd.Series, med_drl: float, med_hc: float) -> tuple[str, str, str]:
    return QMAP[(row["drl_contrib"] >= med_drl, row["hc_contrib"] >= med_hc)]

def main() -> None:
    configure_plot_style()
    inference_df = build_inference_dataframe(tm_ids=range(50))
    convergence_df = build_convergence_dataframe(tm_ids=range(50))
    timing_df = build_timing_breakdown_dataframe(inference_df, convergence_df)

    deep_path = OUTPUT_DIR / "geant2001_deep_analysis.csv"
    if not deep_path.exists():
        raise SystemExit("Thiếu geant2001_deep_analysis.csv. Hãy chạy geant2001_deep_analysis_v3.py trước.")
    deep_df = pd.read_csv(deep_path)

    df = (
        inference_df.merge(deep_df, on="tm_id", how="left", suffixes=("", "_deep"))
        .merge(timing_df, on="tm_id", how="left", suffixes=("", "_timing"))
        .sort_values("tm_id")
        .reset_index(drop=True)
    )
    df["drl_contrib"] = (df["mlu_init"] - df["mlu_drl"]) / df["mlu_init"].clip(lower=1e-9) * 100.0
    df["hc_contrib"] = (df["mlu_drl"] - df["mlu_enero"]) / df["mlu_init"].clip(lower=1e-9) * 100.0
    df["total_improve"] = df["improve_enero"]

    med_drl = float(df["drl_contrib"].median())
    med_hc = float(df["hc_contrib"].median())
    df["q_id"], df["quadrant"], df["q_color"] = zip(*df.apply(assign_quadrant, axis=1, args=(med_drl, med_hc)))

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    for quadrant, group in df.groupby("quadrant"):
        ax.scatter(
            group["drl_contrib"],
            group["hc_contrib"],
            s=110 + 16 * group["hc_only_time_s"].fillna(0.0),
            color=group["q_color"].iloc[0],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
            label=f"{quadrant} (n={len(group)})",
        )
        for row in pd.concat([group.nsmallest(1, "total_improve"), group.nlargest(1, "total_improve")]).drop_duplicates("tm_id").itertuples():
            ax.annotate(f"TM{row.tm_id}", (row.drl_contrib, row.hc_contrib), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(med_drl, color="0.45", linestyle="--", linewidth=1.2)
    ax.axhline(med_hc, color="0.45", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Đóng góp của DRL (% trên OSPF)")
    ax.set_ylabel("Đóng góp của HC (% trên OSPF)")
    ax.set_title("Failure mode map\nKích thước marker tỉ lệ với HC-only runtime")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, "fig_h4a_drl_vs_hc_quadrant")
    plt.close(fig)

    feature_cols = [col for col in ["total_bw", "cv_bw", "entropy", "hub_traffic", "max_bw", "std_bw"] if col in df.columns]
    feature_matrix = df[feature_cols].copy()
    for col in feature_cols:
        col_min = feature_matrix[col].min()
        col_span = feature_matrix[col].max() - col_min
        feature_matrix[col] = 0.0 if col_span < 1e-9 else (feature_matrix[col] - col_min) / col_span
    quadrant_profile = pd.concat([df[["quadrant"]], feature_matrix], axis=1).groupby("quadrant").mean().reindex(sorted(df["quadrant"].unique()))

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    im = ax.imshow(quadrant_profile.to_numpy(), cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(quadrant_profile.index)))
    ax.set_yticklabels(quadrant_profile.index)
    for i in range(quadrant_profile.shape[0]):
        for j in range(quadrant_profile.shape[1]):
            ax.text(j, i, f"{quadrant_profile.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Dấu vết đặc trưng TM theo từng failure mode")
    fig.colorbar(im, ax=ax, shrink=0.92)
    save_figure(fig, "fig_h4b_quadrant_feature_heatmap")
    plt.close(fig)

    timing_summary = (
        df.groupby("quadrant", as_index=False)
        .agg(
            drl_time_s=("cost_drl", "mean"),
            hc_only_time_s=("hc_only_time_s", "mean"),
            enero_time_s=("cost_enero", "mean"),
            hc_steps=("hc_steps", "mean"),
        )
        .sort_values("quadrant")
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    x = np.arange(len(timing_summary))
    width = 0.24
    axes[0].bar(x - width, timing_summary["drl_time_s"], width=width, color=METHOD_COLORS["drl"], label="DRL")
    axes[0].bar(x, timing_summary["hc_only_time_s"], width=width, color=METHOD_COLORS["ls"], label="HC-only")
    axes[0].bar(x + width, timing_summary["enero_time_s"], width=width, color=METHOD_COLORS["enero"], label="Enero")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(timing_summary["quadrant"], rotation=15, ha="right")
    axes[0].set_ylabel("Thời gian trung bình (s)")
    axes[0].set_title("Chi phí thời gian theo failure mode")
    axes[0].grid(True, alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].bar(timing_summary["quadrant"], timing_summary["hc_steps"], color=METHOD_COLORS["ls"], alpha=0.88)
    axes[1].set_ylabel("Số bước HC trung bình")
    axes[1].set_title("Độ sâu HC theo failure mode")
    axes[1].grid(True, alpha=0.25, axis="y")
    save_figure(fig, "fig_h4c_quadrant_timing")
    plt.close(fig)

if __name__ == "__main__":
    main()
