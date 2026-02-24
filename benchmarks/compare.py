#!/usr/bin/env python3
"""
Parse the DATA| lines from bench_faiss.py and the CatWhisper benchmark
binary and print a side-by-side comparison table.

Usage:
    python compare.py catwhisper.txt faiss.txt

Each input file is the captured stdout of the respective benchmark.
DATA lines have the format:
    DATA|<label>|<backend>|<mean_ms>|<median_ms>|<p95_ms>|<p99_ms>|<qps>
"""

import sys
import re
from collections import defaultdict


def parse_file(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith("DATA|"):
                continue
            parts = line.split("|")
            if len(parts) != 8:
                continue
            _, label, backend, mean, median, p95, p99, qps = parts
            rows.append(dict(
                label   = label.strip(),
                backend = backend.strip(),
                mean    = float(mean),
                median  = float(median),
                p95     = float(p95),
                p99     = float(p99),
                qps     = float(qps),
            ))
    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: compare.py <catwhisper_output.txt> <faiss_output.txt>")
        sys.exit(1)

    cw_rows    = parse_file(sys.argv[1])
    faiss_rows = parse_file(sys.argv[2])

    # Index by (label, mode) where mode = single|batch
    def key(row):
        backend = row["backend"]
        if "single" in backend:
            return (row["label"], "single")
        if "batch" in backend:
            return (row["label"], "batch")
        return (row["label"], backend)

    cw_map    = {key(r): r for r in cw_rows}
    faiss_map = defaultdict(dict)
    for r in faiss_rows:
        k = key(r)
        tag = "gpu" if "gpu" in r["backend"] else "cpu"
        faiss_map[k][tag] = r

    configs = sorted({r["label"] for r in cw_rows + faiss_rows})
    modes   = ["single", "batch"]

    print("=" * 90)
    print("CatWhisper vs FAISS — mean latency (ms) / QPS   [lower ms = better]")
    print("=" * 90)

    header = f"{'Config':<14} {'Mode':<7}  {'CW fp16 ms':>12} {'CW fp16 QPS':>12}  {'FAISS-GPU ms':>13} {'FAISS-GPU QPS':>14}  {'Ratio CW/GPU':>13}"
    print(header)
    print("-" * 90)

    for cfg in configs:
        for mode in modes:
            k = (cfg, mode)
            cw  = cw_map.get(k)
            fgpu = faiss_map.get(k, {}).get("gpu")
            fcpu = faiss_map.get(k, {}).get("cpu")

            if cw is None and fgpu is None:
                continue

            cw_ms  = f"{cw['mean']:>12.3f}"  if cw   else f"{'N/A':>12}"
            cw_qps = f"{cw['qps']:>12.1f}"   if cw   else f"{'N/A':>12}"
            fg_ms  = f"{fgpu['mean']:>13.3f}" if fgpu else f"{'N/A':>13}"
            fg_qps = f"{fgpu['qps']:>14.1f}"  if fgpu else f"{'N/A':>14}"

            if cw and fgpu:
                ratio = cw["mean"] / fgpu["mean"]
                ratio_s = f"{ratio:>12.2f}x"
            else:
                ratio_s = f"{'N/A':>13}"

            print(f"{cfg:<14} {mode:<7}  {cw_ms} {cw_qps}  {fg_ms} {fg_qps}  {ratio_s}")

    print()
    print("Ratio CW/GPU > 1 means CatWhisper is slower than FAISS-GPU by that factor.")
    print()

    # Also print FAISS-CPU for reference
    print("-" * 60)
    print("FAISS-CPU reference (single-query):")
    print(f"{'Config':<14} {'Mean ms':>10} {'QPS':>10}")
    print("-" * 38)
    for cfg in configs:
        k = (cfg, "single")
        fcpu = faiss_map.get(k, {}).get("cpu")
        if fcpu:
            print(f"{cfg:<14} {fcpu['mean']:>10.3f} {fcpu['qps']:>10.1f}")


if __name__ == "__main__":
    main()
