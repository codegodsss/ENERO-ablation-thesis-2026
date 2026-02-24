"""
batch_eval.py — Chạy eval trên tất cả topology trong experiments/topos_eval.txt
                và xuất kết quả ra CSV.

Cách dùng (chạy từ thư mục gốc ENERO):
    # Bước 1: Chỉ eval (tạo pckl files) — chạy khi mới train xong model
    python experiments/batch_eval.py --variant baseline --step eval

    # Bước 2: Chỉ export CSV từ pckl đã có
    python experiments/batch_eval.py --variant baseline --step csv

    # Bước 3: Cả hai (eval + export CSV) liên tiếp
    python experiments/batch_eval.py --variant baseline --step all

Tham số thường dùng:
    --variant   : tên thí nghiệm (dùng làm prefix trong tên CSV)
    --logs      : đường dẫn file logs để tìm best model checkpoint
    --f1        : tên folder dataset (mặc định: results_single_top)
    --n         : số processes song song cho eval (mặc định: 2)
    --step      : eval | csv | all (mặc định: all)
"""

import os
import sys
import subprocess
import argparse
import time

TOPOS_FILE = os.path.join(os.path.dirname(__file__), "topos_eval.txt")


def load_topologies(path: str) -> list:
    """Đọc danh sách topology từ file, bỏ qua dòng trống và comment (#)."""
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy file: {path}")
        sys.exit(1)
    topologies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                topologies.append(line)
    return topologies


def run_eval(topo: str, args: argparse.Namespace) -> bool:
    """
    Gọi eval_on_single_topology.py cho 1 topology.
    Trả về True nếu thành công.
    """
    # f2 cần có dạng NEW_{topo}/EVALUATE (theo README gốc)
    f2 = f"NEW_{topo}/EVALUATE"

    cmd = [
        sys.executable, "eval_on_single_topology.py",
        "-max_edge", str(args.max_edge),
        "-min_edge", str(args.min_edge),
        "-max_nodes", str(args.max_nodes),
        "-min_nodes", str(args.min_nodes),
        "-n", str(args.n),
        "-f1", args.f1,
        "-f2", f2,
        "-d", args.logs,
    ]

    print(f"\n{'='*60}")
    print(f"[EVAL] Topology: {topo}")
    print(f"       Command : {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] eval thất bại cho {topo} (returncode={result.returncode})")
        return False
    print(f"[OK] Eval xong {topo} — {elapsed:.1f}s")
    return True


def run_export_csv(topo: str, args: argparse.Namespace) -> bool:
    """
    Gọi experiments/export_csv.py để đọc pckl và ghi ra CSV.
    Trả về True nếu thành công.
    """
    cmd = [
        sys.executable, "experiments/export_csv.py",
        "--topo", topo,
        "--variant", args.variant,
        "--f1", args.f1,
        "--logs", args.logs,
        "--dataset_base", args.dataset_base,
        "--output_dir", args.output_dir,
    ]

    print(f"\n[CSV] Xuất CSV cho topology: {topo}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] export_csv thất bại cho {topo}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch eval trên nhiều topology")
    parser.add_argument(
        "--variant", required=True,
        help="Tên thí nghiệm (dùng làm prefix CSV). Ví dụ: baseline, v1_reward"
    )
    parser.add_argument(
        "--logs", default="./Logs/expSP_3top_15_B_NEWLogs.txt",
        help="Đường dẫn file logs để tìm best model"
    )
    parser.add_argument(
        "--f1", default="results_single_top",
        help="Tên dataset folder (mặc định: results_single_top)"
    )
    parser.add_argument(
        "--max_edge", type=int, default=100,
        help="Số cạnh tối đa topology được eval"
    )
    parser.add_argument(
        "--min_edge", type=int, default=5,
        help="Số cạnh tối thiểu"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=30,
        help="Số nodes tối đa"
    )
    parser.add_argument(
        "--min_nodes", type=int, default=1,
        help="Số nodes tối thiểu"
    )
    parser.add_argument(
        "--n", type=int, default=2,
        help="Số processes song song cho eval"
    )
    parser.add_argument(
        "--dataset_base", default="../Enero_datasets/dataset_sing_top/data",
        help="Thư mục gốc dataset"
    )
    parser.add_argument(
        "--output_dir", default="./results",
        help="Thư mục lưu CSV"
    )
    parser.add_argument(
        "--step", choices=["eval", "csv", "all"], default="all",
        help="eval=chỉ chạy eval; csv=chỉ export CSV; all=cả hai (mặc định: all)"
    )
    parser.add_argument(
        "--topos_file", default=TOPOS_FILE,
        help="Đường dẫn đến file danh sách topology"
    )
    args = parser.parse_args()

    # ── Đọc danh sách topology ────────────────────────────────────────────────
    topologies = load_topologies(args.topos_file)
    print(f"[INFO] Tìm thấy {len(topologies)} topology: {topologies}")
    print(f"[INFO] Variant: '{args.variant}' | Step: '{args.step}'")

    # ── Theo dõi kết quả ──────────────────────────────────────────────────────
    eval_results = {}
    csv_results  = {}

    total_start = time.time()

    for topo in topologies:
        eval_ok = True
        csv_ok  = True

        # ── Bước 1: Eval ──────────────────────────────────────────────────────
        if args.step in ("eval", "all"):
            eval_ok = run_eval(topo, args)
        eval_results[topo] = eval_ok

        # ── Bước 2: Export CSV ────────────────────────────────────────────────
        if args.step in ("csv", "all"):
            if eval_ok:  # chỉ export nếu eval thành công (hoặc mode csv-only)
                csv_ok = run_export_csv(topo, args)
            elif args.step == "all":
                print(f"[SKIP] Bỏ qua export CSV cho {topo} vì eval thất bại.")
                csv_ok = False
            else:  # step == "csv": vẫn cố export dù không biết eval ok không
                csv_ok = run_export_csv(topo, args)
        csv_results[topo] = csv_ok

    # ── Tóm tắt kết quả ───────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT — variant: {args.variant} | total: {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  {'Topology':<25} {'Eval':>8} {'CSV':>8}")
    print(f"  {'-'*45}")
    all_ok = True
    for topo in topologies:
        e_str = "OK" if eval_results.get(topo, True) else "FAIL"
        c_str = "OK" if csv_results.get(topo, True)  else "FAIL"
        if e_str == "FAIL" or c_str == "FAIL":
            all_ok = False
        print(f"  {topo:<25} {e_str:>8} {c_str:>8}")

    print(f"\n  CSV output: {os.path.abspath(args.output_dir)}/")
    if all_ok:
        print("  Tất cả thành công ✓")
    else:
        print("  Có một số bước thất bại — kiểm tra lại log ở trên ↑")


if __name__ == "__main__":
    main()

