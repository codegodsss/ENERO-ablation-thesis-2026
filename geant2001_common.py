#!/usr/bin/env python3
"""Hàm dùng chung cho nhóm script phân tích Geant2001."""
from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TOPOLOGY_NAME = "Geant2001"
N_EVAL_TMS = 50
CAPACITY = 85000.0
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULT_INDEX = {
    "mlu_enero": 3,
    "mlu_drl_sa": 4,
    "n_edges": 6,
    "mlu_ls": 7,
    "mlu_sap": 8,
    "mlu_drl": 9,
    "mlu_init": 11,
    "cost_drl_sa": 12,
    "cost_sap": 13,
    "cost_drl": 14,
    "cost_ls": 15,
    "cost_enero": 16,
}

METHOD_LABELS = {
    "ospf": "OSPF/SP",
    "drl": "DRL only",
    "ls": "LS (HC)",
    "enero": "Enero (DRL+HC)",
}

METHOD_COLORS = {
    "ospf": "#d1495b",
    "drl": "#f28e2b",
    "ls": "#4e79a7",
    "enero": "#2a9d8f",
}

METHOD_MARKERS = {
    "ospf": "^",
    "drl": "o",
    "ls": "s",
    "enero": "D",
}

def find_existing_path(*candidates: str) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None

PATHS = {
    "graph": find_existing_path(
        "~/ENERO-Thesis/Enero_datasets/dataset_sing_top/data/results_single_top/Geant2001/Geant2001.graph",
        "~/ENERO-Thesis/Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/Geant2001/Geant2001.graph",
    ),
    "tm_dir": find_existing_path(
        "~/ENERO-Thesis/Enero_datasets/dataset_sing_top/data/results_single_top/Geant2001/TM",
        "~/ENERO-Thesis/Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/Geant2001/TM",
    ),
    "pckl_dir": find_existing_path(
        "~/ENERO-Thesis/Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_Geant2001/Enero_3top_15_B_NEW/Geant2001",
        "~/ENERO-Thesis/Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_Geant2001/SP_3top_15_B_NEW/Geant2001",
        "~/ENERO-Thesis/Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo/SP_3top_15_B_NEW/Geant2001",
    ),
}

def require_paths(*keys: str) -> tuple[Path, ...]:
    missing = [key for key in keys if PATHS.get(key) is None]
    if missing:
        raise FileNotFoundError(f"Thiếu đường dẫn bắt buộc: {', '.join(missing)}")
    return tuple(PATHS[key] for key in keys)

def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )

def save_figure(fig, stem: str, out_dir: Optional[Path] = None, dpi: int = 180) -> None:
    target_dir = out_dir or OUTPUT_DIR
    target_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(target_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")

def safe_pearsonr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) < 1e-10 or np.std(y_arr) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(x_arr, y_arr)
    return float(r), float(p)

def safe_ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))

def safe_linear_fit(x: Iterable[float], y: Iterable[float]) -> Optional[tuple[float, float]]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.std(x_arr) < 1e-12:
        return None
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    return float(slope), float(intercept)

def parse_graph(filepath: os.PathLike[str] | str) -> nx.DiGraph:
    graph = nx.DiGraph()
    with Path(filepath).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("Link_") or line.startswith("edge_")):
                continue
            parts = line.split()
            src = int(parts[1])
            dst = int(parts[2])
            weight = float(parts[3])
            bw = float(parts[4])
            graph.add_edge(src, dst, weight=weight, capacity=bw)
    return graph

def parse_demands(filepath: os.PathLike[str] | str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(filepath).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 4 or parts[0] == "label":
                continue
            rows.append(
                {
                    "src": int(parts[1]),
                    "dst": int(parts[2]),
                    "bw": float(parts[3]),
                }
            )
    return rows

def load_shortest_paths(path: Optional[os.PathLike[str] | str] = None) -> dict:
    if path is not None and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    graph_file, = require_paths("graph")
    graph = parse_graph(graph_file).to_undirected()
    data: dict[str, list[int]] = {}
    for src in graph.nodes:
        for dst in graph.nodes:
            if src == dst:
                continue
            data[f"{src}:{dst}"] = nx.shortest_path(graph, src, dst, weight="weight")
    return data

def get_shortest_path(sp_data: dict, src: int, dst: int, graph_undirected: nx.Graph) -> list[int]:
    key = f"{src}:{dst}"
    if key in sp_data:
        return list(sp_data[key])
    return nx.shortest_path(graph_undirected, src, dst, weight="weight")

def compute_ospf_link_utilization(
    demands: list[dict[str, float]],
    sp_data: dict,
    graph_undirected: nx.Graph,
    edges: list[tuple[int, int]],
    capacity: float = CAPACITY,
) -> dict[tuple[int, int], float]:
    loads = {edge: 0.0 for edge in edges}
    for demand in demands:
        path = get_shortest_path(sp_data, demand["src"], demand["dst"], graph_undirected)
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            loads[edge] += demand["bw"]
    return {edge: value / capacity for edge, value in loads.items()}

def iter_eval_tm_ids(tm_dir: Optional[os.PathLike[str] | str] = None) -> list[int]:
    directory = Path(tm_dir) if tm_dir else require_paths("tm_dir")[0]
    tm_ids: list[int] = []
    for path in directory.glob(f"{TOPOLOGY_NAME}.*.demands"):
        try:
            tm_ids.append(int(path.stem.split(".")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(tm_ids))

def load_results_array(path: os.PathLike[str] | str) -> Optional[np.ndarray]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data, dtype=object)

def build_demands_dataframe(
    tm_dir: Optional[os.PathLike[str] | str] = None,
    tm_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    directory = Path(tm_dir) if tm_dir else require_paths("tm_dir")[0]
    ids = list(tm_ids) if tm_ids is not None else iter_eval_tm_ids(directory)
    rows: list[dict[str, float]] = []
    for tm_id in ids:
        for demand in parse_demands(directory / f"{TOPOLOGY_NAME}.{tm_id}.demands"):
            rows.append({"tm_id": tm_id, **demand})
    return pd.DataFrame(rows)

def build_tm_stats_dataframe(
    tm_dir: Optional[os.PathLike[str] | str] = None,
    tm_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    demands_df = build_demands_dataframe(tm_dir=tm_dir, tm_ids=tm_ids)
    if demands_df.empty:
        return pd.DataFrame()

    rows = []
    for tm_id, group in demands_df.groupby("tm_id"):
        bw = group["bw"].to_numpy(dtype=float)
        bw_norm = bw / max(bw.sum(), 1e-9)
        top_k = max(1, int(len(bw) * 0.15))
        top = np.sort(bw)[-top_k:]
        rows.append(
            {
                "tm_id": int(tm_id),
                "n_demands": int(len(group)),
                "total_bw": float(bw.sum()),
                "mean_bw": float(bw.mean()),
                "std_bw": float(bw.std()),
                "cv_bw": float(bw.std() / max(bw.mean(), 1e-9)),
                "max_bw": float(bw.max()),
                "entropy": float(-(bw_norm * np.log(bw_norm + 1e-12)).sum()),
                "top15_total_bw": float(top.sum()),
                "top15_mean_bw": float(top.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("tm_id").reset_index(drop=True)

def improvement_pct(baseline: pd.Series | np.ndarray, current: pd.Series | np.ndarray) -> np.ndarray:
    base = np.asarray(baseline, dtype=float)
    cur = np.asarray(current, dtype=float)
    return (base - cur) / np.clip(base, 1e-9, None) * 100.0

def build_inference_dataframe(
    pckl_dir: Optional[os.PathLike[str] | str] = None,
    tm_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    directory = Path(pckl_dir) if pckl_dir else require_paths("pckl_dir")[0]
    ids = list(tm_ids) if tm_ids is not None else range(N_EVAL_TMS)
    rows = []
    for tm_id in ids:
        results = load_results_array(directory / f"{TOPOLOGY_NAME}.{tm_id}.pckl")
        if results is None:
            continue
        row = {"tm_id": int(tm_id)}
        for key, index in RESULT_INDEX.items():
            row[key] = float(results[index]) if index < len(results) else np.nan
        row["improve_drl"] = float(improvement_pct([row["mlu_init"]], [row["mlu_drl"]])[0])
        row["improve_ls"] = float(improvement_pct([row["mlu_init"]], [row["mlu_ls"]])[0])
        row["improve_enero"] = float(improvement_pct([row["mlu_init"]], [row["mlu_enero"]])[0])
        row["delta_drl_to_enero"] = float(row["mlu_drl"] - row["mlu_enero"])
        row["delta_ls_to_enero"] = float(row["mlu_ls"] - row["mlu_enero"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("tm_id").reset_index(drop=True)
    return df

def build_method_long_dataframe(inference_df: pd.DataFrame) -> pd.DataFrame:
    if inference_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"tm_id": int(row.tm_id), "method": "ospf", "mlu": float(row.mlu_init), "runtime_s": 0.0}
            for row in inference_df.itertuples()
        ]
        + [
            {"tm_id": int(row.tm_id), "method": "drl", "mlu": float(row.mlu_drl), "runtime_s": float(row.cost_drl)}
            for row in inference_df.itertuples()
        ]
        + [
            {"tm_id": int(row.tm_id), "method": "ls", "mlu": float(row.mlu_ls), "runtime_s": float(row.cost_ls)}
            for row in inference_df.itertuples()
        ]
        + [
            {"tm_id": int(row.tm_id), "method": "enero", "mlu": float(row.mlu_enero), "runtime_s": float(row.cost_enero)}
            for row in inference_df.itertuples()
        ]
    )

def load_timesteps(path: os.PathLike[str] | str) -> list[dict[str, float]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = []
    for entry in raw:
        row = {
            "time_s": float(entry[0]),
            "mlu": float(entry[1]),
        }
        if len(entry) >= 3:
            row["aux_1"] = float(entry[2])
        if len(entry) >= 4:
            row["drl_target_mlu"] = float(entry[3])
        rows.append(row)
    return rows

def build_convergence_dataframe(
    pckl_dir: Optional[os.PathLike[str] | str] = None,
    tm_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    directory = Path(pckl_dir) if pckl_dir else require_paths("pckl_dir")[0]
    ids = list(tm_ids) if tm_ids is not None else range(N_EVAL_TMS)
    rows = []
    for tm_id in ids:
        points = load_timesteps(directory / f"{TOPOLOGY_NAME}.{tm_id}.timesteps")
        if not points:
            continue
        drl_target = points[0].get("drl_target_mlu", np.nan)
        best_drl_hit = np.nan
        if np.isfinite(drl_target):
            for point in points:
                if point["mlu"] <= drl_target + 1e-10:
                    best_drl_hit = point["time_s"]
                    break
        conv_time = float(points[-1]["time_s"])
        rows.append(
            {
                "tm_id": int(tm_id),
                "n_checkpoints": int(len(points)),
                "conv_time": conv_time,
                "best_mlu": float(min(point["mlu"] for point in points)),
                "drl_target_mlu": float(drl_target) if np.isfinite(drl_target) else np.nan,
                "best_drl_hit_time_s": float(best_drl_hit) if np.isfinite(best_drl_hit) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("tm_id").reset_index(drop=True)

def build_timing_breakdown_dataframe(
    inference_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> pd.DataFrame:
    if inference_df.empty:
        return pd.DataFrame()
    df = inference_df.merge(convergence_df, on="tm_id", how="left", suffixes=("", "_conv"))
    df["best_drl_hit_time_s"] = df["best_drl_hit_time_s"].fillna(df["cost_drl"])
    df["best_drl_hit_time_s"] = np.minimum(df["best_drl_hit_time_s"], df["cost_drl"])
    df["drl_tail_after_best_s"] = (df["cost_drl"] - df["best_drl_hit_time_s"]).clip(lower=0.0)
    df["hc_only_time_s"] = (df["cost_enero"] - df["cost_drl"]).clip(lower=0.0)

    def infer_hc_steps(row: pd.Series) -> int:
        if not np.isfinite(row.get("drl_target_mlu", np.nan)):
            return int(max(row.get("n_checkpoints", 1) - 1, 0))
        path_points = load_timesteps(require_paths("pckl_dir")[0] / f"{TOPOLOGY_NAME}.{int(row.tm_id)}.timesteps")
        if not path_points:
            return 0
        return int(sum(point["mlu"] < row["drl_target_mlu"] - 1e-10 for point in path_points))

    df["hc_steps"] = df.apply(infer_hc_steps, axis=1)
    df["time_per_hc_step_s"] = np.where(
        df["hc_steps"] > 0,
        df["hc_only_time_s"] / df["hc_steps"],
        0.0,
    )
    keep = [
        "tm_id",
        "cost_drl",
        "cost_ls",
        "cost_enero",
        "conv_time",
        "best_drl_hit_time_s",
        "drl_tail_after_best_s",
        "hc_only_time_s",
        "hc_steps",
        "time_per_hc_step_s",
    ]
    return df[keep].sort_values("tm_id").reset_index(drop=True)
