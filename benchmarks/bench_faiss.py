#!/usr/bin/env python3
"""
FAISS CPU + GPU benchmark for all index types.

Uses CLUSTERED DATA for IVF/PQ/HNSW (matching CatWhisper benchmarks).
Uses RANDOM DATA for IndexFlat (matching CatWhisper benchmarks).

Usage:
    /home/ire/anaconda3/envs/bench/bin/python bench_faiss.py [--large] [--index TYPE]
"""

import sys
import time
import math
import argparse
import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: faiss not found. Activate the 'bench' environment:")
    print("  conda activate bench")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration - matches CatWhisper C++ benchmarks exactly
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(n_vectors=10_000,   dim=128, k=10, label="10K  x128",
         nlist=32,  nprobe=16, m=16, nbits=8, M_hnsw=16, ef_construction=100),
    dict(n_vectors=100_000,  dim=128, k=10, label="100K x128",
         nlist=64,  nprobe=16, m=16, nbits=8, M_hnsw=16, ef_construction=100),
    dict(n_vectors=100_000,  dim=256, k=10, label="100K x256",
         nlist=64,  nprobe=16, m=16, nbits=8, M_hnsw=16, ef_construction=100),
]
LARGE_CONFIG = dict(n_vectors=1_000_000, dim=128, k=10, label="1M   x128",
                    nlist=256, nprobe=32, m=16, nbits=8, M_hnsw=32, ef_construction=200)

N_WARMUP = 20
N_BENCH  = 200
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data generation - matches CatWhisper C++ benchmarks exactly
# ---------------------------------------------------------------------------

def generate_random_vectors(n: int, dim: int, seed: int = RNG_SEED) -> np.ndarray:
    """Random N(0,1) vectors - for IndexFlat."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def generate_clustered_vectors(n: int, dim: int, n_clusters: int, seed: int = RNG_SEED) -> np.ndarray:
    """
    Clustered vectors - for IVF/PQ/HNSW.
    Matches CatWhisper's generate_clustered_vectors() exactly:
    - Cluster centers: N(0, 10)
    - Noise: N(0, 0.5)
    """
    rng = np.random.default_rng(seed)
    
    # Generate cluster centers: N(0, 10)
    centers = rng.normal(0.0, 10.0, (n_clusters, dim)).astype(np.float32)
    
    # Assign each vector to a random cluster
    cluster_ids = rng.integers(0, n_clusters, size=n)
    
    # Generate vectors: center + N(0, 0.5) noise
    noise = rng.normal(0.0, 0.5, (n, dim)).astype(np.float32)
    data = centers[cluster_ids] + noise
    
    return data


def generate_queries(n: int, dim: int, seed: int = 123) -> np.ndarray:
    """Random queries - same for all tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def stats(times_ms: list) -> dict:
    a = sorted(times_ms)
    n = len(a)
    return dict(
        mean   = sum(a) / n,
        median = a[n // 2],
        p95    = a[int(n * 0.95)],
        p99    = a[int(n * 0.99)],
        qps    = 1000.0 / (sum(a) / n),
    )


def bench_index(index, queries: np.ndarray, k: int) -> dict:
    """Time single-query search."""
    # warmup
    for q in queries[:N_WARMUP]:
        index.search(q.reshape(1, -1), k)

    times_ms = []
    for q in queries[N_WARMUP:N_WARMUP + N_BENCH]:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return stats(times_ms)


def compute_recall(index, data: np.ndarray, queries: np.ndarray, k: int, n_check: int = 10) -> float:
    """Compute recall@k vs brute-force ground truth."""
    flat = faiss.IndexFlatL2(data.shape[1])
    flat.add(data)

    total_recall = 0.0
    for i in range(n_check):
        q = queries[N_WARMUP + i:N_WARMUP + i + 1]
        _, gt_ids = flat.search(q, k)
        _, result_ids = index.search(q, k)

        gt_set = set(gt_ids[0])
        hits = sum(1 for rid in result_ids[0] if rid in gt_set)
        total_recall += hits / k

    return total_recall / n_check


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def make_ivf_flat(data: np.ndarray, cfg: dict, gpu_res=None):
    n, dim = data.shape
    nlist = cfg["nlist"]
    nprobe = cfg["nprobe"]

    if gpu_res is not None:
        quantizer = faiss.IndexFlatL2(dim)
        quantizer_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, quantizer)
        index = faiss.GpuIndexIVFFlat(gpu_res, quantizer_gpu, dim, nlist, faiss.METRIC_L2)
        index.nprobe = nprobe
        index.train(data)
        index.add(data)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.nprobe = nprobe
        index.train(data)
        index.add(data)

    return index


def make_ivf_pq(data: np.ndarray, cfg: dict, gpu_res=None):
    n, dim = data.shape
    nlist = cfg["nlist"]
    nprobe = cfg["nprobe"]
    m = cfg["m"]
    nbits = cfg["nbits"]

    if gpu_res is not None:
        quantizer = faiss.IndexFlatL2(dim)
        quantizer_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, quantizer)
        index = faiss.GpuIndexIVFPQ(gpu_res, quantizer_gpu, dim, nlist, m, nbits, faiss.METRIC_L2)
        index.nprobe = nprobe
        index.train(data)
        index.add(data)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_L2)
        index.nprobe = nprobe
        index.train(data)
        index.add(data)

    return index


def make_hnsw(data: np.ndarray, cfg: dict):
    dim = data.shape[1]
    M = cfg["M_hnsw"]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = cfg["ef_construction"]
    index.add(data)

    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="Include 1M-vector config")
    parser.add_argument("--index", choices=["flat", "ivf", "pq", "hnsw", "all"], default="all")
    args = parser.parse_args()

    configs = CONFIGS + ([LARGE_CONFIG] if args.large else [])

    print("=" * 70)
    print("FAISS Benchmark (Clustered Data for IVF/PQ/HNSW)")
    print("=" * 70)
    print(f"FAISS version : {faiss.__version__}")

    gpu_available = hasattr(faiss, "StandardGpuResources")
    if gpu_available:
        try:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)
            print("GPU support   : YES")
        except Exception as e:
            print(f"GPU support   : FAILED ({e})")
            gpu_available = False
    else:
        print("GPU support   : NO (faiss-cpu)")

    print(f"Warmup queries: {N_WARMUP}")
    print(f"Bench queries : {N_BENCH}")
    print()

    results = []

    for cfg in configs:
        n = cfg["n_vectors"]
        dim = cfg["dim"]
        k = cfg["k"]
        lbl = cfg["label"]
        nlist = cfg["nlist"]

        print("-" * 70)
        print(f"Config: {lbl}  k={k}")

        # Generate queries (always random)
        queries = generate_queries(N_WARMUP + N_BENCH, dim)

        # ----------------------------------------------------------------
        # IndexFlat - RANDOM DATA (matches CatWhisper)
        # ----------------------------------------------------------------
        if args.index in ["flat", "all"]:
            print(f"\n  [IndexFlatL2] (random data)")
            data = generate_random_vectors(n, dim)

            flat_idx = faiss.IndexFlatL2(dim)
            flat_idx.add(data)

            s = bench_index(flat_idx, queries, k)
            print(f"    CPU:  median {s['median']:.3f} ms, mean {s['mean']:.3f} ms, QPS {s['qps']:.0f}")
            results.append((lbl, "faiss-cpu-flat", s, -1.0))

            if gpu_available:
                gpu_flat = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dim))
                gpu_flat.add(data)
                s = bench_index(gpu_flat, queries, k)
                print(f"    GPU:  median {s['median']:.3f} ms, mean {s['mean']:.3f} ms, QPS {s['qps']:.0f}")
                results.append((lbl, "faiss-gpu-flat", s, -1.0))

        # ----------------------------------------------------------------
        # IndexIVFFlat - CLUSTERED DATA (matches CatWhisper)
        # ----------------------------------------------------------------
        if args.index in ["ivf", "all"]:
            print(f"\n  [IndexIVFFlat] nlist={nlist}, nprobe={cfg['nprobe']} (clustered data)")
            data = generate_clustered_vectors(n, dim, nlist)

            try:
                ivf = make_ivf_flat(data, cfg, gpu_res=None)
                s = bench_index(ivf, queries, k)
                rec = compute_recall(ivf, data, queries, k)
                print(f"    CPU:  median {s['median']:.3f} ms, QPS {s['qps']:.0f}, recall {rec:.1%}")
                results.append((lbl, "faiss-cpu-ivf", s, rec))
            except Exception as e:
                print(f"    CPU FAILED: {e}")

            if gpu_available:
                try:
                    gpu_ivf = make_ivf_flat(data, cfg, gpu_res=res)
                    s = bench_index(gpu_ivf, queries, k)
                    rec = compute_recall(gpu_ivf, data, queries, k)
                    print(f"    GPU:  median {s['median']:.3f} ms, QPS {s['qps']:.0f}, recall {rec:.1%}")
                    results.append((lbl, "faiss-gpu-ivf", s, rec))
                except Exception as e:
                    print(f"    GPU FAILED: {e}")

        # ----------------------------------------------------------------
        # IndexIVFPQ - CLUSTERED DATA (matches CatWhisper)
        # ----------------------------------------------------------------
        if args.index in ["pq", "all"]:
            print(f"\n  [IndexIVFPQ] nlist={nlist}, nprobe={cfg['nprobe']}, m={cfg['m']} (clustered data)")
            data = generate_clustered_vectors(n, dim, nlist)

            try:
                pq = make_ivf_pq(data, cfg, gpu_res=None)
                s = bench_index(pq, queries, k)
                rec = compute_recall(pq, data, queries, k)
                print(f"    CPU:  median {s['median']:.3f} ms, QPS {s['qps']:.0f}, recall {rec:.1%}")
                results.append((lbl, "faiss-cpu-pq", s, rec))
            except Exception as e:
                print(f"    CPU FAILED: {e}")

            if gpu_available:
                try:
                    gpu_pq = make_ivf_pq(data, cfg, gpu_res=res)
                    s = bench_index(gpu_pq, queries, k)
                    rec = compute_recall(gpu_pq, data, queries, k)
                    print(f"    GPU:  median {s['median']:.3f} ms, QPS {s['qps']:.0f}, recall {rec:.1%}")
                    results.append((lbl, "faiss-gpu-pq", s, rec))
                except Exception as e:
                    print(f"    GPU FAILED: {e}")

        # ----------------------------------------------------------------
        # IndexHNSW - CLUSTERED DATA (matches CatWhisper)
        # ----------------------------------------------------------------
        if args.index in ["hnsw", "all"]:
            print(f"\n  [IndexHNSW] M={cfg['M_hnsw']} (clustered data, CPU only)")
            data = generate_clustered_vectors(n, dim, nlist)

            try:
                hnsw = make_hnsw(data, cfg)

                for ef_search in [10, 50, 100]:
                    hnsw.hnsw.efSearch = ef_search
                    s = bench_index(hnsw, queries, k)
                    rec = compute_recall(hnsw, data, queries, k)
                    print(f"    CPU ef={ef_search}: median {s['median']:.3f} ms, QPS {s['qps']:.0f}, recall {rec:.1%}")
                    results.append((lbl, f"faiss-cpu-hnsw-ef{ef_search}", s, rec))
            except Exception as e:
                print(f"    FAILED: {e}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<14} {'Backend':<24} {'Median':>8} {'QPS':>8} {'Recall':>7}")
    print("-" * 70)
    for lbl, backend, s, rec in results:
        rec_str = f"{rec:.1%}" if rec >= 0 else "-"
        print(f"{lbl:<14} {backend:<24} {s['median']:>8.3f} {s['qps']:>8.0f} {rec_str:>7}")

    print()
    print("# BENCHMARK_DATA_BEGIN")
    for lbl, backend, s, rec in results:
        print(f"DATA|{lbl}|{backend}|{s['mean']:.4f}|{s['median']:.4f}|{s['p95']:.4f}|{s['p99']:.4f}|{s['qps']:.2f}|{rec:.4f}")
    print("# BENCHMARK_DATA_END")


if __name__ == "__main__":
    main()
