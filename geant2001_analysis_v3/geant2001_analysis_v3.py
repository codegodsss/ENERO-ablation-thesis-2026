#!/usr/bin/env python3
"""Phân tích tổng hợp Geant2001 và xuất bảng CSV chuẩn hóa."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geant2001_common import (
    CAPACITY,
    N_EVAL_TMS,
    OUTPUT_DIR,
    TOPOLOGY_NAME,
    build_convergence_dataframe,
    build_demands_dataframe,
    build_inference_dataframe,
    build_method_long_dataframe,
    build_timing_breakdown_dataframe,
    build_tm_stats_dataframe,
    parse_graph,
    require_paths,
)

def summarize_topology(graph_directed: nx.DiGraph, graph_undirected: nx.Graph) -> dict[str, float]:
    degree_values = np.asarray([degree for _, degree in graph_undirected.degree()], dtype=float)
    node_bc = nx.betweenness_centrality(graph_undirected, normalized=True)
    hub_node = max(node_bc, key=node_bc.get)

    print(f"Nodes: {graph_undirected.number_of_nodes()}")
    print(f"Undirected edges: {graph_undirected.number_of_edges()}")
    print(f"Link capacity: {CAPACITY:.0f}")
    print(f"Density: {nx.density(graph_undirected):.4f}")
    if nx.is_connected(graph_undirected):
        diameter = nx.diameter(graph_undirected)
        avg_path_length = nx.average_shortest_path_length(graph_undirected)
        print(f"Diameter: {diameter}")
        print(f"Avg shortest path length: {avg_path_length:.3f}")
    else:
        diameter = None
        avg_path_length = None
        print("Diameter: N/A")
        print("Avg shortest path length: N/A")
    print(
        "Degree stats: "
        f"min={degree_values.min():.0f}, "
        f"max={degree_values.max():.0f}, "
        f"mean={degree_values.mean():.2f}, "
        f"std={degree_values.std():.2f}"
    )
    print(f"Hub node: {hub_node} (betweenness={node_bc[hub_node]:.4f})")

    return {
        "n_nodes": int(graph_undirected.number_of_nodes()),
        "n_edges_directed": int(graph_directed.number_of_edges()),
        "n_edges_undirected": int(graph_undirected.number_of_edges()),
        "density": float(nx.density(graph_undirected)),
        "diameter": diameter,
        "avg_path_length": avg_path_length,
        "avg_clustering": float(nx.average_clustering(graph_undirected)),
        "hub_node": int(hub_node),
        "hub_betweenness": float(node_bc[hub_node]),
    }

def main() -> None:
    graph_file, tm_dir, pckl_dir = require_paths("graph", "tm_dir", "pckl_dir")

    print("\n" + "=" * 64)
    print(f"PHÂN TÍCH {TOPOLOGY_NAME}")
    print("=" * 64)

    graph_directed = parse_graph(graph_file)
    graph_undirected = graph_directed.to_undirected()
    topology_stats = summarize_topology(graph_directed, graph_undirected)
    with (OUTPUT_DIR / "geant2001_topology_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_stats, handle, indent=2)

    demands_df = build_demands_dataframe(tm_dir=tm_dir, tm_ids=range(N_EVAL_TMS))
    tm_stats_df = build_tm_stats_dataframe(tm_dir=tm_dir, tm_ids=range(N_EVAL_TMS))
    inference_df = build_inference_dataframe(pckl_dir=pckl_dir, tm_ids=range(N_EVAL_TMS))
    convergence_df = build_convergence_dataframe(pckl_dir=pckl_dir, tm_ids=range(N_EVAL_TMS))
    timing_df = build_timing_breakdown_dataframe(inference_df, convergence_df)
    method_long_df = build_method_long_dataframe(inference_df)

    pair_stats_df = (
        demands_df.groupby(["src", "dst"], as_index=False)
        .agg(
            bw_mean=("bw", "mean"),
            bw_std=("bw", "std"),
            bw_min=("bw", "min"),
            bw_max=("bw", "max"),
            bw_p95=("bw", lambda s: float(np.quantile(s, 0.95))),
        )
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )

    inference_df.to_csv(OUTPUT_DIR / "geant2001_inference_results.csv", index=False)
    convergence_df.to_csv(OUTPUT_DIR / "geant2001_convergence.csv", index=False)
    tm_stats_df.to_csv(OUTPUT_DIR / "geant2001_tm_stats.csv", index=False)
    pair_stats_df.to_csv(OUTPUT_DIR / "geant2001_demand_pairs.csv", index=False)
    method_long_df.to_csv(OUTPUT_DIR / "geant2001_method_metrics_long.csv", index=False)
    timing_df.to_csv(OUTPUT_DIR / "geant2001_timing_breakdown.csv", index=False)

    print(f"TMs đã đọc: {len(inference_df)}/{N_EVAL_TMS}")
    print(f"Số dòng demand: {len(demands_df):,}")
    print(
        "MLU trung bình: "
        f"OSPF={inference_df['mlu_init'].mean():.4f}, "
        f"DRL={inference_df['mlu_drl'].mean():.4f}, "
        f"LS={inference_df['mlu_ls'].mean():.4f}, "
        f"Enero={inference_df['mlu_enero'].mean():.4f}"
    )
    print(
        "Thời gian trung bình: "
        f"DRL={inference_df['cost_drl'].mean():.2f}s, "
        f"LS={inference_df['cost_ls'].mean():.2f}s, "
        f"Enero={inference_df['cost_enero'].mean():.2f}s"
    )
    if not timing_df.empty:
        print(
            "Breakdown trung bình: "
            f"best DRL hit={timing_df['best_drl_hit_time_s'].mean():.2f}s, "
            f"đuôi DRL={timing_df['drl_tail_after_best_s'].mean():.2f}s, "
            f"HC-only={timing_df['hc_only_time_s'].mean():.2f}s"
        )

if __name__ == "__main__":
    main()
