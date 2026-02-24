"""
export_csv.py — Đọc kết quả .pckl từ eval_on_single_topology.py và xuất ra CSV.

Cách dùng (chạy từ thư mục gốc ENERO):
    python experiments/export_csv.py \
        --topo EliBackbone \
        --variant baseline \
        --f1 results_single_top \
        --logs ./Logs/expSP_3top_15_B_NEWLogs.txt

Kết quả được lưu tại: results/{variant}_{topo}.csv

Các cột CSV:
    variant          : tên thí nghiệm (baseline, v1_reward, v2_demands, ...)
    topology         : tên topology (EliBackbone, ...)
    tm_id            : traffic matrix id (0..49)
    ospf_init        : max link utilization trước khi optimize (OSPF/SP baseline)
    drl_only         : max link util sau DRL (trước local search)
    hill_only        : max link util dùng hill climbing thuần (không DRL)
    enero            : max link util sau DRL + hill climbing (kết quả cuối của ENERO)
    drl_time_s       : thời gian DRL (giây)
    hill_time_s      : thời gian hill climbing (giây)
    total_time_s     : tổng thời gian (giây)
    num_edges        : số edges của topology
"""

import os
import sys
import pickle
import csv
import glob
import argparse


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_differentiation_str(logs_path: str) -> str:
    """Trích xuất differentiation_str từ tên file logs (khớp với eval_on_single_topology.py)."""
    aux = logs_path.split(".")
    aux = aux[1].split("exp") if len(aux) > 1 else logs_path.split("exp")
    return str(aux[1].split("Logs")[0]) if len(aux) > 1 else "SP_3top_15_B_NEW"


def find_topo_name_in_eval_dir(eval_dir: str) -> str:
    """
    Trong thư mục evalRes_NEW_{topo}/EVALUATE/{diff_str}/ có thư mục con
    tên chính là topology name (ví dụ 'EliBackbone'). Trả về tên đó.
    Nếu có nhiều, trả về cái đầu tiên.
    """
    subdirs = [
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"Không tìm thấy thư mục con trong: {eval_dir}")
    if len(subdirs) > 1:
        print(f"[WARN] Có nhiều topology trong {eval_dir}: {subdirs}. Dùng '{subdirs[0]}'.")
    return subdirs[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Xuất kết quả eval sang CSV")
    parser.add_argument(
        "--topo", required=True,
        help="Tên topology (không có tiền tố NEW_). Ví dụ: EliBackbone"
    )
    parser.add_argument(
        "--variant", required=True,
        help="Tên variant thí nghiệm. Ví dụ: baseline, v1_reward_composite"
    )
    parser.add_argument(
        "--f1", default="results_single_top",
        help="Tên dataset folder (giống tham số -f1 của eval_on_single_topology.py)"
    )
    parser.add_argument(
        "--logs", default="./Logs/expSP_3top_15_B_NEWLogs.txt",
        help="Đường dẫn tới file logs (giống tham số -d)"
    )
    parser.add_argument(
        "--dataset_base", default="../Enero_datasets/dataset_sing_top/data",
        help="Thư mục gốc chứa dataset (mặc định: ../Enero_datasets/dataset_sing_top/data)"
    )
    parser.add_argument(
        "--output_dir", default="./results",
        help="Thư mục lưu CSV output (mặc định: ./results)"
    )
    args = parser.parse_args()

    # ── Tìm differentiation_str từ file logs ──────────────────────────────────
    diff_str = get_differentiation_str(args.logs)
    print(f"[INFO] differentiation_str = '{diff_str}'")

    # ── Xây dựng đường dẫn tới thư mục chứa .pckl ────────────────────────────
    # eval_on_single_topology.py lưu pckl tại:
    #   {dataset_base}/{f1}/evalRes_NEW_{topo}/EVALUATE/{diff_str}/{topo_name}/
    eval_res_base = os.path.join(
        args.dataset_base,
        args.f1,
        f"evalRes_NEW_{args.topo}",
        "EVALUATE",
        diff_str
    )

    if not os.path.exists(eval_res_base):
        print(f"[ERROR] Không tìm thấy thư mục kết quả eval: {eval_res_base}")
        print("        Hãy chạy eval_on_single_topology.py trước.")
        sys.exit(1)

    # Lấy tên topology thực tế (tên thư mục con, ví dụ 'EliBackbone')
    topo_name = find_topo_name_in_eval_dir(eval_res_base)
    pckl_dir = os.path.join(eval_res_base, topo_name)
    pckl_files = sorted(glob.glob(os.path.join(pckl_dir, "*.pckl")))

    if not pckl_files:
        print(f"[ERROR] Không có file .pckl nào trong {pckl_dir}")
        sys.exit(1)

    print(f"[INFO] Tìm thấy {len(pckl_files)} file .pckl trong {pckl_dir}")

    # ── Đọc pckl và tổng hợp ─────────────────────────────────────────────────
    # Index trong mảng results[] (xem script_eval_on_single_topology.py):
    #   results[3]  = max_link_uti_DRL_SP_HILL  → ENERO (DRL + Hill)
    #   results[7]  = max_link_uti_sp_hill_climb → Hill climbing only
    #   results[9]  = max_link_uti_DRL_SP        → DRL only
    #   results[11] = OSPF_init                  → trước khi optimize
    #   results[14] = optim_cost_DRL_GNN         → thời gian DRL (s)
    #   results[15] = optim_cost_HILL            → thời gian hill climb (s)
    #   results[16] = tổng thời gian (s)
    #   results[6]  = số edges

    rows = []
    skipped = 0
    for fpath in pckl_files:
        fname = os.path.basename(fpath)
        # Tên file: {topo_name}.{tm_id}.pckl
        parts = fname.replace(".pckl", "").split(".")
        tm_id = parts[-1] if len(parts) >= 2 else "?"

        try:
            with open(fpath, "rb") as f:
                results = pickle.load(f)

            rows.append({
                "variant":      args.variant,
                "topology":     args.topo,
                "tm_id":        tm_id,
                "ospf_init":    round(float(results[11]), 6),
                "drl_only":     round(float(results[9]),  6),
                "hill_only":    round(float(results[7]),  6),
                "enero":        round(float(results[3]),  6),
                "drl_time_s":   round(float(results[14]), 4),
                "hill_time_s":  round(float(results[15]), 4),
                "total_time_s": round(float(results[16]), 4),
                "num_edges":    int(results[6]),
            })
        except Exception as e:
            print(f"[WARN] Bỏ qua {fname}: {e}")
            skipped += 1

    if not rows:
        print("[ERROR] Không đọc được bất kỳ kết quả nào.")
        sys.exit(1)

    # ── Ghi CSV ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.variant}_{args.topo}.csv")

    fieldnames = [
        "variant", "topology", "tm_id",
        "ospf_init", "drl_only", "hill_only", "enero",
        "drl_time_s", "hill_time_s", "total_time_s", "num_edges"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Đã ghi {len(rows)} hàng → {csv_path}")
    if skipped:
        print(f"[WARN] Bỏ qua {skipped} file lỗi.")

    # ── In tóm tắt nhanh ──────────────────────────────────────────────────────
    import statistics
    enero_vals = [r["enero"] for r in rows]
    ospf_vals  = [r["ospf_init"] for r in rows]
    print(f"\n  Topology  : {args.topo} | Variant: {args.variant}")
    print(f"  TM count  : {len(rows)}")
    print(f"  OSPF init : mean={statistics.mean(ospf_vals):.4f}  max={max(ospf_vals):.4f}")
    print(f"  ENERO     : mean={statistics.mean(enero_vals):.4f}  max={max(enero_vals):.4f}")
    improvement = (statistics.mean(ospf_vals) - statistics.mean(enero_vals)) / statistics.mean(ospf_vals) * 100
    print(f"  Cải thiện : {improvement:.2f}%  (mean OSPF → mean ENERO)")


if __name__ == "__main__":
    main()

