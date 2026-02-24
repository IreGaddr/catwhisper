#!/usr/bin/env python3
"""
FAISS CPU + GPU benchmark.

Measures add() and single-query search() latency across a set of
configurations that match the CatWhisper C++ benchmark so results are
directly comparable.

Usage:
    /path/to/anaconda3/envs/bench/bin/python bench_faiss.py [--large]

Flags:
    --large    Include 1M-vector configuration (needs ~512 MB GPU memory
               and ~2 GB host RAM).
"""

import sys
import time
import math
import argparse
import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: faiss not found.  Activate the 'bench' conda environment:")
    print("  conda activate bench   # or use the full interpreter path")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(n_vectors=10_000,   dim=128, k=10, label="10K  x128"),
    dict(n_vectors=100_000,  dim=128, k=10, label="100K x128"),
    dict(n_vectors=100_000,  dim=256, k=10, label="100K x256"),
]
LARGE_CONFIG = dict(n_vectors=1_000_000, dim=128, k=10, label="1M   x128")

N_WARMUP = 20    # queries discarded for warmup
N_BENCH  = 200   # queries timed

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stats(times_ms: list[float]) -> dict:
    a = sorted(times_ms)
    n = len(a)
    return dict(
        mean   = sum(a) / n,
        median = a[n // 2],
        p95    = a[int(n * 0.95)],
        p99    = a[int(n * 0.99)],
        qps    = 1000.0 / (sum(a) / n),
    )


def print_stats(label: str, s: dict) -> None:
    print(f"    {label}")
    print(f"      Mean:   {s['mean']:>8.3f} ms")
    print(f"      Median: {s['median']:>8.3f} ms")
    print(f"      P95:    {s['p95']:>8.3f} ms")
    print(f"      P99:    {s['p99']:>8.3f} ms")
    print(f"      QPS:    {s['qps']:>8.1f}")


def generate_data(n: int, dim: int, seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data    = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((N_WARMUP + N_BENCH, dim)).astype(np.float32)
    return data, queries


def bench_index(index, queries: np.ndarray, k: int) -> dict:
    """Time single-query search N_WARMUP + N_BENCH times, discard warmup."""
    all_queries = queries
    # warmup
    for q in all_queries[:N_WARMUP]:
        index.search(q.reshape(1, -1), k)

    times_ms = []
    for q in all_queries[N_WARMUP:]:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return stats(times_ms)


def bench_batch(index, queries: np.ndarray, k: int) -> dict:
    """Time batched search over all N_BENCH queries at once (10 repetitions)."""
    q_batch = queries[N_WARMUP:]
    # warmup
    for _ in range(3):
        index.search(q_batch, k)

    rep = 10
    times_ms = []
    for _ in range(rep):
        t0 = time.perf_counter()
        index.search(q_batch, k)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0 / N_BENCH)   # per-query

    return stats(times_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Include 1M-vector configuration")
    args = parser.parse_args()

    configs = CONFIGS + ([LARGE_CONFIG] if args.large else [])

    # --- FAISS version and GPU info ----------------------------------------
    print("=" * 62)
    print("FAISS Benchmark")
    print("=" * 62)
    print(f"FAISS version : {faiss.__version__}")

    gpu_available = hasattr(faiss, "StandardGpuResources")
    if gpu_available:
        try:
            res = faiss.StandardGpuResources()
            # Touch it to verify initialization
            res.setTempMemory(128 * 1024 * 1024)  # 128 MB temp
            print("GPU support   : YES (StandardGpuResources OK)")
        except Exception as e:
            print(f"GPU support   : FAILED ({e})")
            gpu_available = False
    else:
        print("GPU support   : NO (faiss-cpu build)")

    print(f"Warmup queries: {N_WARMUP}")
    print(f"Bench queries : {N_BENCH}")
    print()

    results = []   # (label, backend, single_stats, batch_stats)

    for cfg in configs:
        n   = cfg["n_vectors"]
        dim = cfg["dim"]
        k   = cfg["k"]
        lbl = cfg["label"]

        print("-" * 62)
        print(f"Config: {lbl}  k={k}")
        print(f"  Generating {n:,} x {dim} floats … ", end="", flush=True)
        data, queries = generate_data(n, dim)
        print("done")

        # ----------------------------------------------------------------
        # FAISS CPU
        # ----------------------------------------------------------------
        cpu_idx = faiss.IndexFlatL2(dim)

        t0 = time.perf_counter()
        cpu_idx.add(data)
        add_cpu_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [CPU] Add: {add_cpu_ms:.1f} ms  ({n / add_cpu_ms * 1000:.0f} vec/s)")

        single_cpu = bench_index(cpu_idx, queries, k)
        batch_cpu  = bench_batch(cpu_idx, queries, k)
        print_stats("CPU single-query", single_cpu)
        print_stats("CPU batch/query ", batch_cpu)
        results.append((lbl, "faiss-cpu-single", single_cpu))
        results.append((lbl, "faiss-cpu-batch",  batch_cpu))

        # ----------------------------------------------------------------
        # FAISS GPU
        # ----------------------------------------------------------------
        if gpu_available:
            try:
                gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)

                # Re-time add on GPU (data is copied during index_cpu_to_gpu,
                # so time a fresh transfer by creating from scratch).
                fresh_cpu = faiss.IndexFlatL2(dim)
                t0 = time.perf_counter()
                fresh_cpu.add(data)
                gpu_idx2 = faiss.index_cpu_to_gpu(res, 0, fresh_cpu)
                add_gpu_ms = (time.perf_counter() - t0) * 1000.0
                print(f"  [GPU] Add+transfer: {add_gpu_ms:.1f} ms  ({n / add_gpu_ms * 1000:.0f} vec/s)")

                single_gpu = bench_index(gpu_idx2, queries, k)
                batch_gpu  = bench_batch(gpu_idx2, queries, k)
                print_stats("GPU single-query", single_gpu)
                print_stats("GPU batch/query ", batch_gpu)
                results.append((lbl, "faiss-gpu-single", single_gpu))
                results.append((lbl, "faiss-gpu-batch",  batch_gpu))

            except Exception as e:
                print(f"  [GPU] FAILED: {e}")

        print()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=" * 62)
    print("SUMMARY — mean search latency (ms) / QPS")
    print("=" * 62)
    print(f"{'Config':<14} {'Backend':<22} {'Mean ms':>8} {'QPS':>8}")
    print("-" * 56)
    for (lbl, backend, s) in results:
        print(f"{lbl:<14} {backend:<22} {s['mean']:>8.3f} {s['qps']:>8.1f}")
    print()

    # Machine-readable output for compare script
    print("# BENCHMARK_DATA_BEGIN")
    for (lbl, backend, s) in results:
        print(f"DATA|{lbl}|{backend}|{s['mean']:.4f}|{s['median']:.4f}|{s['p95']:.4f}|{s['p99']:.4f}|{s['qps']:.2f}")
    print("# BENCHMARK_DATA_END")


if __name__ == "__main__":
    main()
