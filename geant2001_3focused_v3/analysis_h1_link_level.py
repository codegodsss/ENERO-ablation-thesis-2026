#!/usr/bin/env python3
"""Phân tích mức liên kết cho Geant2001."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geant2001_common import (
    CAPACITY,
    METHOD_COLORS,
    build_inference_dataframe,
    compute_ospf_link_utilization,
    configure_plot_style,
    load_shortest_paths,
    parse_demands,
    parse_graph,
    require_paths,
    safe_linear_fit,
    safe_pearsonr,
    save_figure,
)

def main() -> None:
    configure_plot_style()
    graph_file, tm_dir, pckl_dir = require_paths("graph", "tm_dir", "pckl_dir")
    graph = parse_graph(graph_file).to_undirected()
    sp_data = load_shortest_paths()
    inference_df = build_inference_dataframe(pckl_dir=pckl_dir, tm_ids=range(50)).set_index("tm_id")

    edges = sorted(tuple(sorted((u, v))) for u, v in graph.edges())
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    edge_labels = [f"{u}-{v}" for u, v in edges]
    ospf_util = np.zeros((50, len(edges)), dtype=float)
    drl_util = np.zeros_like(ospf_util)
    ls_util = np.zeros_like(ospf_util)
    enero_util = np.zeros_like(ospf_util)

    for tm_id in range(50):
        demands = parse_demands(tm_dir / f"Geant2001.{tm_id}.demands")
        base_util = compute_ospf_link_utilization(demands, sp_data, graph, edges, CAPACITY)
        for edge, value in base_util.items():
            ospf_util[tm_id, edge_index[edge]] = value
        if tm_id not in inference_df.index:
            continue
        row = inference_df.loc[tm_id]
        init_mlu = max(float(row["mlu_init"]), 1e-9)
        drl_util[tm_id] = ospf_util[tm_id] * (float(row["mlu_drl"]) / init_mlu)
        ls_util[tm_id] = ospf_util[tm_id] * (float(row["mlu_ls"]) / init_mlu)
        enero_util[tm_id] = ospf_util[tm_id] * (float(row["mlu_enero"]) / init_mlu)

    edge_bc = nx.edge_betweenness_centrality(graph, normalized=True)
    edge_bc_arr = np.asarray([edge_bc.get(edge, edge_bc.get((edge[1], edge[0]), 0.0)) for edge in edges], dtype=float)
    order = np.argsort(ospf_util.mean(axis=0) + 0.35 * edge_bc_arr)[::-1]

    fig, axes = plt.subplots(4, 1, figsize=(15.5, 13.0), sharex=True, constrained_layout=True)
    heatmaps = [
        (ospf_util[:, order].T, "OSPF/SP (chính xác)"),
        (drl_util[:, order].T, "DRL only (ước lượng theo tỷ lệ MLU)"),
        (ls_util[:, order].T, "LS/HC (ước lượng theo tỷ lệ MLU)"),
        (enero_util[:, order].T, "Enero (ước lượng theo tỷ lệ MLU)"),
    ]
    vmax = max(np.quantile(ospf_util, 0.98), 1.0)
    image = None
    for ax, (matrix, title) in zip(axes, heatmaps):
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=vmax)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Rank link")
    axes[-1].set_xlabel("Traffic matrix ID")
    ticks = np.linspace(0, len(order) - 1, min(8, len(order)), dtype=int)
    axes[-1].set_yticks(ticks)
    axes[-1].set_yticklabels([edge_labels[order[idx]] for idx in ticks])
    fig.colorbar(image, ax=axes, label="Utilization", shrink=0.94)
    save_figure(fig, "fig_h1a_link_util_heatmap")
    plt.close(fig)

    mean_ospf = ospf_util.mean(axis=0)
    relief_enero = mean_ospf - enero_util.mean(axis=0)
    relief_ls = mean_ospf - ls_util.mean(axis=0)
    rank_df = pd.DataFrame(
        {
            "edge": edge_labels,
            "ospf_mean": mean_ospf,
            "edge_betweenness": edge_bc_arr,
            "enero_relief": relief_enero,
            "ls_relief": relief_ls,
        }
    ).sort_values(["ospf_mean", "edge_betweenness"], ascending=False)

    top_links = rank_df.head(12)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    x = np.arange(len(top_links))
    width = 0.36
    ax.bar(x - width / 2, top_links["ls_relief"], width=width, color=METHOD_COLORS["ls"], label="LS relief")
    ax.bar(x + width / 2, top_links["enero_relief"], width=width, color=METHOD_COLORS["enero"], label="Enero relief")
    ax.plot(x, top_links["edge_betweenness"], color="black", marker="o", linewidth=1.6, label="Edge betweenness")
    ax.set_xticks(x)
    ax.set_xticklabels(top_links["edge"], rotation=35, ha="right")
    ax.set_ylabel("Giảm tải / độ trung gian")
    ax.set_title("Những link nóng nhất và mức giảm tải đạt được")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    save_figure(fig, "fig_h1b_relief_top_links")
    plt.close(fig)

    pearson_r, pearson_p = safe_pearsonr(rank_df["edge_betweenness"], rank_df["enero_relief"])
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(rank_df["edge_betweenness"], rank_df["enero_relief"], s=75, color=METHOD_COLORS["enero"], alpha=0.85)
    fit = safe_linear_fit(rank_df["edge_betweenness"], rank_df["enero_relief"])
    if fit is not None:
        slope, intercept = fit
        x_grid = np.linspace(rank_df["edge_betweenness"].min(), rank_df["edge_betweenness"].max(), 200)
        ax.plot(x_grid, slope * x_grid + intercept, color="black", linewidth=1.8)
    ax.set_xlabel("Edge betweenness")
    ax.set_ylabel("Mức giảm tải trung bình của Enero")
    ax.set_title(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}")
    ax.grid(True, alpha=0.25)
    save_figure(fig, "fig_h1c_betweenness_vs_relief")
    plt.close(fig)

if __name__ == "__main__":
    main()
