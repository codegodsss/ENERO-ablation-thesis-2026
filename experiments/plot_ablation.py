"""
plot_ablation.py — Đọc tất cả CSV trong results/ và vẽ boxplot so sánh các variants.

Cách dùng (chạy từ thư mục gốc ENERO):
    python experiments/plot_ablation.py

    # Chỉ so sánh một số variant cụ thể:
    python experiments/plot_ablation.py --variants baseline v1_reward_composite

    # Chỉ vẽ một topology:
    python experiments/plot_ablation.py --topos EliBackbone Janetbackbone

    # Đổi thư mục đọc CSV và lưu plot:
    python experiments/plot_ablation.py --input_dir ./results --output_dir ./plots

Output:
    plots/ablation_enero.png         — Boxplot max link util của ENERO (DRL+Hill)
    plots/ablation_drl_only.png      — Boxplot của DRL only (trước local search)
    plots/ablation_vs_ospf.png       — % Cải thiện so với OSPF init
    plots/ablation_time.png          — Thời gian tối ưu (giây)
    plots/ablation_summary.csv       — Bảng mean±std tóm tắt tất cả variants
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Không cần GUI — tương thích server/headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────────────────
# Màu sắc và style
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#2196F3",  # blue     — baseline
    "#F44336",  # red      — v1
    "#4CAF50",  # green    — v2
    "#FF9800",  # orange   — v3
    "#9C27B0",  # purple   — v4
    "#00BCD4",  # cyan     — v5
    "#FF5722",  # deep orange
    "#607D8B",  # blue grey
]


# ──────────────────────────────────────────────────────────────────────────────
# Load CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_results(input_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    # Bỏ qua ablation_summary.csv nếu có
    csv_files = [f for f in csv_files if "summary" not in os.path.basename(f)]

    if not csv_files:
        print(f"[ERROR] Không có file CSV nào trong {input_dir}")
        sys.exit(1)

    frames = []
    for fpath in sorted(csv_files):
        try:
            df = pd.read_csv(fpath)
            frames.append(df)
            print(f"[INFO] Đọc {os.path.basename(fpath)} — {len(df)} hàng")
        except Exception as e:
            print(f"[WARN] Bỏ qua {fpath}: {e}")

    if not frames:
        print("[ERROR] Không đọc được file CSV nào.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Boxplot chung
# ──────────────────────────────────────────────────────────────────────────────

def make_grouped_boxplot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
    variants: list,
    topologies: list,
):
    """
    Vẽ grouped boxplot: trục X = topology, mỗi nhóm gồm N_variant boxes (màu khác nhau).
    """
    n_topo    = len(topologies)
    n_variant = len(variants)
    group_gap = 1.0             # khoảng cách giữa các nhóm topology
    box_width = 0.6 / n_variant # mỗi box hẹp lại theo số variants

    fig, ax = plt.subplots(figsize=(max(10, n_topo * n_variant * 1.2), 6))

    xtick_pos, xtick_labels = [], []

    for t_idx, topo in enumerate(topologies):
        group_center = t_idx * (1.0 + group_gap)

        for v_idx, variant in enumerate(variants):
            offset  = (v_idx - (n_variant - 1) / 2) * (box_width + 0.05)
            pos     = group_center + offset
            color   = PALETTE[v_idx % len(PALETTE)]

            data = df[(df["topology"] == topo) & (df["variant"] == variant)][metric].dropna().values

            if len(data) == 0:
                continue

            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                notch=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", markerfacecolor=color, markersize=3, alpha=0.5),
            )

        # Label giữa mỗi nhóm topology
        xtick_pos.append(group_center)
        xtick_labels.append(topo)

    # Legend
    legend_patches = [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], alpha=0.7, label=v)
        for i, v in enumerate(variants)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def make_summary_csv(df: pd.DataFrame, variants: list, topologies: list, output_path: str):
    """Tạo bảng mean ± std cho từng (variant, topology, metric)."""
    metrics = ["ospf_init", "drl_only", "hill_only", "enero", "total_time_s"]
    records = []

    for variant in variants:
        for topo in topologies:
            sub = df[(df["variant"] == variant) & (df["topology"] == topo)]
            if sub.empty:
                continue
            row = {"variant": variant, "topology": topo, "n_tm": len(sub)}
            for m in metrics:
                vals = sub[m].dropna()
                row[f"{m}_mean"] = round(vals.mean(), 5) if len(vals) > 0 else None
                row[f"{m}_std"]  = round(vals.std(),  5) if len(vals) > 0 else None
            # % improvement ENERO vs OSPF
            ospf = sub["ospf_init"].dropna()
            enero= sub["enero"].dropna()
            if len(ospf) > 0 and len(enero) > 0:
                row["improvement_pct"] = round(
                    (ospf.mean() - enero.mean()) / ospf.mean() * 100, 2
                )
            records.append(row)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")

    # In ra console cho tiện
    print("\n  === TÓM TẮT KẾT QUẢ ===")
    cols_show = ["variant", "topology", "n_tm", "ospf_init_mean", "enero_mean", "improvement_pct", "total_time_s_mean"]
    cols_available = [c for c in cols_show if c in summary_df.columns]
    print(summary_df[cols_available].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vẽ ablation plot từ CSV results")
    parser.add_argument(
        "--input_dir", default="./results",
        help="Thư mục chứa CSV (mặc định: ./results)"
    )
    parser.add_argument(
        "--output_dir", default="./plots",
        help="Thư mục lưu plot (mặc định: ./plots)"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Chỉ vẽ các variant này (mặc định: tất cả có trong CSV)"
    )
    parser.add_argument(
        "--topos", nargs="+", default=None,
        help="Chỉ vẽ các topology này (mặc định: tất cả có trong CSV)"
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_results(args.input_dir)

    # Lọc variants và topologies theo yêu cầu
    all_variants   = sorted(df["variant"].unique().tolist())
    all_topologies = sorted(df["topology"].unique().tolist())

    variants   = args.variants   if args.variants   else all_variants
    topologies = args.topos      if args.topos       else all_topologies

    # Kiểm tra hợp lệ
    for v in variants:
        if v not in all_variants:
            print(f"[WARN] Variant '{v}' không có trong CSV. Có: {all_variants}")
    for t in topologies:
        if t not in all_topologies:
            print(f"[WARN] Topology '{t}' không có trong CSV. Có: {all_topologies}")

    df = df[df["variant"].isin(variants) & df["topology"].isin(topologies)]

    print(f"\n[INFO] Variants   : {variants}")
    print(f"[INFO] Topologies : {topologies}")
    print(f"[INFO] Tổng hàng  : {len(df)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Plot 1: ENERO (DRL + Hill) — metric chính ────────────────────────────
    make_grouped_boxplot(
        df=df,
        metric="enero",
        ylabel="Max Link Utilization",
        title="ENERO (DRL + Local Search) — Max Link Utilization theo Topology",
        output_path=os.path.join(args.output_dir, "ablation_enero.png"),
        variants=variants,
        topologies=topologies,
    )

    # ── Plot 2: DRL only ──────────────────────────────────────────────────────
    make_grouped_boxplot(
        df=df,
        metric="drl_only",
        ylabel="Max Link Utilization",
        title="DRL only (trước Local Search)",
        output_path=os.path.join(args.output_dir, "ablation_drl_only.png"),
        variants=variants,
        topologies=topologies,
    )

    # ── Plot 3: % Improvement (ENERO vs OSPF) ────────────────────────────────
    df["improvement_pct"] = (df["ospf_init"] - df["enero"]) / df["ospf_init"] * 100

    make_grouped_boxplot(
        df=df,
        metric="improvement_pct",
        ylabel="% Cải thiện so với OSPF init",
        title="% Cải thiện Max Link Utilization (OSPF → ENERO)",
        output_path=os.path.join(args.output_dir, "ablation_vs_ospf.png"),
        variants=variants,
        topologies=topologies,
    )

    # ── Plot 4: Thời gian tối ưu ─────────────────────────────────────────────
    make_grouped_boxplot(
        df=df,
        metric="total_time_s",
        ylabel="Thời gian (giây)",
        title="Tổng thời gian tối ưu (DRL + Hill Climbing)",
        output_path=os.path.join(args.output_dir, "ablation_time.png"),
        variants=variants,
        topologies=topologies,
    )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    make_summary_csv(
        df=df,
        variants=variants,
        topologies=topologies,
        output_path=os.path.join(args.output_dir, "ablation_summary.csv"),
    )


if __name__ == "__main__":
    main()

