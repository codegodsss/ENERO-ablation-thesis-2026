#!/usr/bin/env python3
"""Phân tích và tóm tắt kết quả eval của 3 topology mới."""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

def parse_results(results_folder: str) -> tuple[pd.DataFrame, set[str]]:
    result_files = sorted(Path(results_folder).glob("*.pkl"))
    print(f"Tìm thấy {len(result_files)} file kết quả trong {results_folder}")
    if not result_files:
        return pd.DataFrame(), set()

    rows = []
    topologies: set[str] = set()
    for path in result_files:
        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
        except Exception as exc:
            print(f"Lỗi khi đọc {path}: {exc}")
            continue
        rows.append(data)
        topologies.add(data["topology"])
    return pd.DataFrame(rows), topologies

def summarize_by_topology(df: pd.DataFrame, topologies: set[str]) -> pd.DataFrame:
    summary_rows = []
    print("\n" + "=" * 80)
    print("TÓM TẮT THEO TOPOLOGY")
    print("=" * 80)
    for topology in sorted(topologies):
        topo_df = df[df["topology"] == topology]
        row = {
            "Topology": topology,
            "Episodes": int(len(topo_df)),
            "Avg Reward": float(topo_df["reward"].mean()),
            "Std Reward": float(topo_df["reward"].std()),
            "Avg Error Links": float(topo_df["error_links"].mean()),
            "Avg Max Util": float(topo_df["max_link_uti"].mean()),
            "Avg Min Util": float(topo_df["min_link_uti"].mean()),
            "Avg Util Std": float(topo_df["uti_std"].mean()),
        }
        summary_rows.append(row)
        print(pd.Series(row).to_string())
        print("-" * 80)
    return pd.DataFrame(summary_rows)

def overall_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TỔNG HỢP TOÀN BỘ")
    print("=" * 80)
    print(f"Tổng số episode: {len(df)}")
    print(f"Số topology: {df['topology'].nunique()}")
    print(f"Reward mean/std: {df['reward'].mean():.4f} / {df['reward'].std():.4f}")
    print(f"Max util mean: {df['max_link_uti'].mean():.4f}")
    print(f"Min util mean: {df['min_link_uti'].mean():.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích kết quả eval trên topology mới")
    parser.add_argument("-r", help="thư mục chứa file *.pkl", type=str, required=True, nargs="+")
    args = parser.parse_args()

    results_folder = args.r[0]
    if not os.path.exists(results_folder):
        raise SystemExit(f"Không tồn tại thư mục: {results_folder}")

    df, topologies = parse_results(results_folder)
    if df.empty:
        raise SystemExit("Không có dữ liệu để phân tích.")

    summary_df = summarize_by_topology(df, topologies)
    overall_summary(df)
    print("\nBảng so sánh:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
