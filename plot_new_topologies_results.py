#!/usr/bin/env python3
"""Vẽ boxplot và CDF từ kết quả eval trên 3 topology mới."""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_results(results_folder: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_folder).glob("*.pkl")):
        with path.open("rb") as handle:
            rows.append(pickle.load(handle))
    if not rows:
        raise RuntimeError("Không tìm thấy file .pkl")
    df = pd.DataFrame(rows)
    df["max_link_uti_ratio"] = df["max_link_uti"].astype(float)
    return df

def plot_boxplot_max_uti(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(6.6, 4.2))
    sns.boxplot(x="topology", y="max_link_uti_ratio", data=df, palette="mako")
    plt.xlabel("Topology")
    plt.ylabel("Maximum Link Utilization")
    plt.title("Boxplot MLU theo topology")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_cdf_max_uti(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(6.6, 4.2))
    for topology in sorted(df["topology"].unique()):
        values = np.sort(df[df["topology"] == topology]["max_link_uti_ratio"].to_numpy(dtype=float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        plt.step(values, cdf, where="post", label=topology)
    plt.xlabel("Maximum Link Utilization")
    plt.ylabel("CDF")
    plt.title("CDF MLU theo topology")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ boxplot và CDF từ eval result")
    parser.add_argument("-r", "--results", type=str, required=True, help="thư mục chứa *.pkl")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./Images/EVALUATION/Enero_3top_15_B_NEW_custom",
        help="thư mục lưu ảnh",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.results)
    plot_boxplot_max_uti(df, os.path.join(args.outdir, "Boxplot_MaxLinkUtil_per_Topology.pdf"))
    plot_cdf_max_uti(df, os.path.join(args.outdir, "CDF_MaxLinkUtil_per_Topology.pdf"))
    print("Đã lưu ảnh vào", args.outdir)

if __name__ == "__main__":
    main()
