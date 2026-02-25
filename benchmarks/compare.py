#!/usr/bin/env python3
"""
Parse the DATA| lines from bench_faiss.py and the CatWhisper benchmark
binary and print a side-by-side comparison table.

Usage:
    python compare.py catwhisper.txt faiss.txt [--index TYPE]

Each input file is the captured stdout of the respective benchmark.
DATA lines have the format:
    DATA|<label>|<backend>|<mean_ms>|<median_ms>|<p95_ms>|<p99_ms>|<qps>|<recall>
"""

import sys
import argparse
from collections import defaultdict


def parse_file(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith("DATA|"):
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            _, label, backend, mean, median, p95, p99, qps = parts[:9]
            recall = float(parts[8]) if len(parts) > 8 else -1.0
            rows.append(dict(
                label   = label.strip(),
                backend = backend.strip(),
                mean    = float(mean),
                median  = float(median),
                p95     = float(p95),
                p99     = float(p99),
                qps     = float(qps),
                recall  = recall,
            ))
    return rows


def detect_index_type(backend: str) -> str:
    """Detect index type from backend name."""
    backend_lower = backend.lower()
    if "hnsw" in backend_lower:
        return "hnsw"
    elif "pq" in backend_lower:
        return "pq"
    elif "ivf" in backend_lower:
        return "ivf"
    else:
        return "flat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catwhisper_file", help="CatWhisper benchmark output")
    parser.add_argument("faiss_file", help="FAISS benchmark output")
    parser.add_argument("--index", choices=["flat", "ivf", "pq", "hnsw", "all"],
                        default="all", help="Compare specific index type")
    args = parser.parse_args()

    cw_rows    = parse_file(args.catwhisper_file)
    faiss_rows = parse_file(args.faiss_file)

    # Index by (label, index_type, mode)
    def key(row):
        backend = row["backend"]
        index_type = detect_index_type(backend)
        if "single" in backend:
            mode = "single"
        elif "batch" in backend:
            mode = "batch"
        else:
            mode = "single"
        return (row["label"], index_type, mode)

    cw_map = {key(r): r for r in cw_rows}
    faiss_map = defaultdict(lambda: defaultdict(dict))
    for r in faiss_rows:
        k = key(r)
        tag = "gpu" if "gpu" in r["backend"].lower() else "cpu"
        faiss_map[k[:2]][tag][r["backend"]] = r  # key by (label, index_type)

    configs = sorted({r["label"] for r in cw_rows + faiss_rows})
    index_types = ["flat", "ivf", "pq", "hnsw"] if args.index == "all" else [args.index]

    print("=" * 100)
    print("CatWhisper vs FAISS — latency (ms) / QPS / Recall   [lower ms = better, higher recall = better]")
    print("=" * 100)

    for idx_type in index_types:
        print(f"\n--- {idx_type.upper()} ---")

        header = f"{'Config':<14} {'CW ms':>8} {'CW QPS':>8} {'CW Rec':>7}  {'FAISS-GPU ms':>12} {'GPU QPS':>8} {'GPU Rec':>7}  {'FAISS-CPU ms':>12} {'CPU QPS':>8} {'CPU Rec':>7}"
        print(header)
        print("-" * 100)

        for cfg in configs:
            k = (cfg, idx_type)
            faiss_by_backend = faiss_map.get(k, {})

            # Get best FAISS-GPU and FAISS-CPU for this config
            faiss_gpu = None
            faiss_cpu = None

            for backend, row in faiss_by_backend.get("gpu", {}).items():
                if faiss_gpu is None or row["mean"] < faiss_gpu["mean"]:
                    faiss_gpu = row

            for backend, row in faiss_by_backend.get("cpu", {}).items():
                if faiss_cpu is None or row["mean"] < faiss_cpu["mean"]:
                    faiss_cpu = row

            # For CatWhisper, try to find a matching row
            cw_key = (cfg, idx_type, "single")
            cw = cw_map.get(cw_key)

            # Format output
            cw_ms = f"{cw['mean']:>8.3f}" if cw else f"{'N/A':>8}"
            cw_qps = f"{cw['qps']:>8.1f}" if cw else f"{'N/A':>8}"
            cw_rec = f"{cw['recall']:>6.1%}" if cw and cw['recall'] >= 0 else f"{'N/A':>7}"

            fg_ms = f"{faiss_gpu['mean']:>12.3f}" if faiss_gpu else f"{'N/A':>12}"
            fg_qps = f"{faiss_gpu['qps']:>8.1f}" if faiss_gpu else f"{'N/A':>8}"
            fg_rec = f"{faiss_gpu['recall']:>6.1%}" if faiss_gpu and faiss_gpu['recall'] >= 0 else f"{'N/A':>7}"

            fc_ms = f"{faiss_cpu['mean']:>12.3f}" if faiss_cpu else f"{'N/A':>12}"
            fc_qps = f"{faiss_cpu['qps']:>8.1f}" if faiss_cpu else f"{'N/A':>8}"
            fc_rec = f"{faiss_cpu['recall']:>6.1%}" if faiss_cpu and faiss_cpu['recall'] >= 0 else f"{'N/A':>7}"

            print(f"{cfg:<14} {cw_ms} {cw_qps} {cw_rec}  {fg_ms} {fg_qps} {fg_rec}  {fc_ms} {fc_qps} {fc_rec}")

    print()
    print("CW = CatWhisper, GPU = FAISS-GPU, CPU = FAISS-CPU")
    print("Rec = Recall@10 (higher is better)")
    print()

    # Speedup summary
    print("-" * 60)
    print("Speedup Summary (CatWhisper vs FAISS-GPU):")
    print(f"{'Config':<14} {'Index':<8} {'CW/GPU Ratio':>12} {'Winner':>10}")
    print("-" * 50)

    for idx_type in index_types:
        for cfg in configs:
            k = (cfg, idx_type)
            faiss_by_backend = faiss_map.get(k, {})

            # Get best FAISS-GPU
            faiss_gpu = None
            for backend, row in faiss_by_backend.get("gpu", {}).items():
                if faiss_gpu is None or row["mean"] < faiss_gpu["mean"]:
                    faiss_gpu = row

            cw_key = (cfg, idx_type, "single")
            cw = cw_map.get(cw_key)

            if cw and faiss_gpu:
                ratio = cw["mean"] / faiss_gpu["mean"]
                winner = "CW" if ratio < 1.0 else "FAISS"
                print(f"{cfg:<14} {idx_type:<8} {ratio:>12.2f}x {winner:>10}")


if __name__ == "__main__":
    main()
