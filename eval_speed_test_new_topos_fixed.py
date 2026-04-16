#!/usr/bin/env python3
"""Đo nhanh tốc độ eval trên 3 topology mới."""
from __future__ import annotations

import argparse
import os
import subprocess
import time

TOPOS_TO_TEST = ["Geant2001", "Garr199904", "HiberniaUk"]
DATASET_BASE = "../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo"

def load_best_model_id(log_path: str) -> int:
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in reversed(handle.readlines()):
                parts = line.split(":")
                if parts[0] == "MAX REWD":
                    return int(parts[2].split(",")[0])
    except Exception as exc:
        print(f"Không đọc được log file: {exc}. Dùng model_id = 0")
    return 0

def run_one(topology: str, tm_id: int, model_id: int, diff_str: str) -> float:
    dataset_folder = os.path.join(DATASET_BASE, topology)
    output_dir = f"/tmp/eval_speed_{topology}_{int(time.time())}/"
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        "python3 script_eval_on_new_topologies_full.py "
        f"-t {tm_id} -m {model_id} -g {topology} "
        f"-o {output_dir} -d {diff_str} -f {dataset_folder}"
    )
    start = time.time()
    subprocess.call([cmd], shell=True)
    return time.time() - start

def main() -> None:
    parser = argparse.ArgumentParser(description="Đo tốc độ eval trên 3 topology mới")
    parser.add_argument("-d", help="đường dẫn log file để lấy best model ID", type=str, required=True)
    parser.add_argument("-k", help="số episode test cho mỗi topology", type=int, default=3)
    args = parser.parse_args()

    aux = args.d.split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])
    model_id = load_best_model_id(args.d)

    timings = []
    print("\n" + "=" * 80)
    print("BẮT ĐẦU ĐO TỐC ĐỘ EVAL")
    print("=" * 80)
    for topology in TOPOS_TO_TEST:
        topo_times = []
        for tm_id in range(args.k):
            elapsed = run_one(topology, tm_id, model_id, differentiation_str)
            topo_times.append(elapsed)
            print(f"{topology} | TM {tm_id} | {elapsed:.2f}s")
        timings.extend({"topology": topology, "time_s": value} for value in topo_times)
        print(f"Trung bình {topology}: {sum(topo_times) / len(topo_times):.2f}s")

    total = sum(item["time_s"] for item in timings)
    mean_time = total / max(len(timings), 1)
    print("-" * 80)
    print(f"Tổng số lần đo: {len(timings)}")
    print(f"Thời gian trung bình mỗi episode: {mean_time:.2f}s")
    print(f"Ước tính 450 episode: {mean_time * 450 / 60:.1f} phút")

if __name__ == "__main__":
    main()
