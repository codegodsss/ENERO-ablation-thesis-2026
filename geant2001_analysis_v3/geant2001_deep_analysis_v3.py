#!/usr/bin/env python3
"""Phân tích sâu theo từng traffic matrix của Geant2001."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    parse_demands,
    parse_graph,
    require_paths,
    safe_linear_fit,
    safe_pearsonr,
    save_figure,
)

def compute_tm_features(demands: list[dict[str, float]], hub_node: int) -> dict[str, float]:
    bw = np.asarray([item["bw"] for item in demands], dtype=float)
    bw_norm = bw / max(bw.sum(), 1e-9)
    top_k = max(1, int(len(bw) * 0.15))
    top = np.sort(bw)[-top_k:]
    hub_traffic = sum(item["bw"] for item in demands if item["src"] == hub_node or item["dst"] == hub_node)
    return {
        "total_bw": float(bw.sum()),
        "mean_bw": float(bw.mean()),
        "std_bw": float(bw.std()),
        "cv_bw": float(bw.std() / max(bw.mean(), 1e-9)),
        "max_bw": float(bw.max()),
        "entropy": float(-(bw_norm * np.log(bw_norm + 1e-12)).sum()),
        "top15_total_bw": float(top.sum()),
        "top15_mean_bw": float(top.mean()),
        "hub_traffic": float(hub_traffic),
    }

def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return 0.0
    r, _ = spearmanr(x_arr[mask], y_arr[mask])
    return float(r) if np.isfinite(r) else 0.0

def main() -> None:
    configure_plot_style()
    graph_file, tm_dir, pckl_dir = require_paths("graph", "tm_dir", "pckl_dir")
    graph = parse_graph(graph_file).to_undirected()
    hub_node = max(nx.betweenness_centrality(graph, normalized=True), key=nx.betweenness_centrality(graph).get)

    feature_rows = []
    for tm_id in range(50):
        demands = parse_demands(tm_dir / f"Geant2001.{tm_id}.demands")
        if not demands:
            continue
        feature_rows.append({"tm_id": tm_id, **compute_tm_features(demands, hub_node)})
    feature_df = pd.DataFrame(feature_rows).sort_values("tm_id").reset_index(drop=True)

    inference_df = build_inference_dataframe(pckl_dir=pckl_dir, tm_ids=range(50))
    convergence_df = build_convergence_dataframe(pckl_dir=pckl_dir, tm_ids=range(50))
    timing_df = build_timing_breakdown_dataframe(inference_df, convergence_df)

    deep_df = (
        feature_df.merge(inference_df, on="tm_id", how="inner")
        .merge(timing_df, on="tm_id", how="left", suffixes=("", "_timing"))
        .sort_values("tm_id")
        .reset_index(drop=True)
    )

    corr_rows = []
    for feature in [
        "total_bw",
        "mean_bw",
        "std_bw",
        "cv_bw",
        "max_bw",
        "entropy",
        "top15_total_bw",
        "top15_mean_bw",
        "hub_traffic",
    ]:
        pearson_r, pearson_p = safe_pearsonr(deep_df[feature], deep_df["improve_enero"])
        corr_rows.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": safe_spearman(deep_df[feature], deep_df["improve_enero"]),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r")

    deep_df.to_csv(OUTPUT_DIR / "geant2001_deep_analysis.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "geant2001_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(
        deep_df["total_bw"],
        deep_df["improve_enero"],
        s=90,
        c=deep_df["cv_bw"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.6,
    )
    fit = safe_linear_fit(deep_df["total_bw"], deep_df["improve_enero"])
    if fit is not None:
        slope, intercept = fit
        x_grid = np.linspace(deep_df["total_bw"].min(), deep_df["total_bw"].max(), 200)
        ax.plot(x_grid, slope * x_grid + intercept, color=METHOD_COLORS["enero"], linewidth=2.0)
    ax.set_xlabel("Tổng băng thông TM")
    ax.set_ylabel("Cải thiện của Enero so với OSPF (%)")
    ax.set_title("Mối liên hệ giữa tổng tải TM và mức cải thiện")
    ax.grid(True, alpha=0.25)
    save_figure(fig, "fig_deep_total_bw_vs_gain")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    heatmap_df = corr_df.set_index("feature")[["pearson_r", "spearman_r"]]
    im = ax.imshow(heatmap_df.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(heatmap_df.shape[1]))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(range(heatmap_df.shape[0]))
    ax.set_yticklabels(heatmap_df.index)
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            ax.text(j, i, f"{heatmap_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Tương quan đặc trưng TM với mức cải thiện của Enero")
    fig.colorbar(im, ax=ax, shrink=0.92)
    save_figure(fig, "fig_deep_correlations")
    plt.close(fig)

    best5 = deep_df.nlargest(5, "improve_enero")[["tm_id", "improve_enero", "total_bw", "cv_bw", "entropy"]]
    worst5 = deep_df.nsmallest(5, "improve_enero")[["tm_id", "improve_enero", "total_bw", "cv_bw", "entropy"]]
    print("Top 5 TM tốt nhất:")
    print(best5.to_string(index=False))
    print("\nTop 5 TM khó nhất:")
    print(worst5.to_string(index=False))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
