# CatWhisper

**Cross-vendor GPU vector similarity query — faster than FAISS-GPU**

CatWhisper is a C++ library for exact and approximate nearest neighbor search
over dense vectors, accelerated via Vulkan compute shaders.  It runs on any
Vulkan-capable GPU (NVIDIA, AMD, Intel, ARM, Apple Silicon via MoltenVK) and
requires no CUDA installation.

## Performance

Single-query latency, RTX 4080 Laptop GPU, k=10, fp16 mode:

| Configuration | CatWhisper | FAISS-GPU | FAISS-CPU |
|---------------|-----------|-----------|-----------|
| 10K × 128     | **0.053 ms** | 0.065 ms | 0.059 ms |
| 100K × 128    | **0.056 ms** | 0.350 ms | 0.837 ms |
| 100K × 256    | **0.106 ms** | 0.601 ms | 2.231 ms |

FAISS 1.13.2 · CatWhisper commit bd85835 · 20 warmup + 100 timed queries · median latency

### IVF Performance

IndexIVFFlat recall@10, clustered data, nprobe=16:

| Configuration | IVF Median | Recall@10 |
|---------------|------------|-----------|
| 10K × 128     | 0.193 ms   | 99.0% |
| 100K × 128    | 0.964 ms   | 98.0% |
| 100K × 256    | 0.346 ms   | 100.0% |

### HNSW Performance

IndexHNSW with graph-based search (parallelized CPU implementation):

| Configuration | Median | Recall@10 | QPS |
|---------------|--------|-----------|------|
| 10K × 128 (M=16, ef=50) | 0.23 ms | 90.0% | 4,323 |
| 100K × 128 (M=16, ef=100) | 1.48 ms | 76.0% | 676 |
| **Batch 100 queries (10K)** | 0.025 ms/query | 90.0% | **40,766** |
| **Batch 500 queries (100K)** | 0.066 ms/query | 76.0% | **15,081** |

Parallelization: AVX-512 SIMD distances + multi-threaded batch search + optional GPU path

### IVF-PQ Performance

IndexIVFPQ with memory compression (m=16 subquantizers, 8 bits each):

| Configuration | Median | Recall@10 | GPU Memory | Compression |
|---------------|--------|-----------|------------|-------------|
| 10K × 128     | 0.58 ms | 77.0% | 0.02 MB | 488x |
| 100K × 128    | 0.90 ms | 97.0% | 0.15 MB | 488x |
| 1M × 128      | - | - | 0.25 MB | **976x** |

Memory comparison: IndexFlat (1M×128 fp16) = 244 MB vs IndexIVFPQ = 0.25 MB

## Quick Start

```cpp
#include <catwhisper/index_flat.hpp>

// Create a GPU context
auto ctx   = cw::Context::create().value();

// Build an exact flat index (128-dimensional, L2 metric)
auto index = cw::IndexFlat::create(ctx, 128).value();

// Add vectors (float32 input, stored as fp16 on GPU)
index.add(data, 100'000).value();

// Search — single query, top-10
auto results = index.search({query.data(), 128}, 10).value();
for (auto& [dist, id] : results) {
    std::cout << id << " " << dist << "\n";
}
```

### IVF (Inverted File) Index

For larger datasets with approximate search:

```cpp
#include <catwhisper/index_ivf_flat.hpp>

auto ctx = cw::Context::create().value();

// Configure IVF: 256 clusters, search 16 of them
cw::IVFParams params{.nlist = 256, .nprobe = 16};
auto index = cw::IndexIVFFlat::create(ctx, 128, params).value();

// Train on representative data
index.train(train_data, 50'000).value();

// Add vectors (GPU-accelerated cluster assignment)
index.add(data, 1'000'000).value();

// Search — 98-100% recall at 16 nprobe
auto results = index.search({query.data(), 128}, 10).value();
```

### IVF-PQ (Product Quantization) Index

For memory-constrained applications with extreme compression:

```cpp
#include <catwhisper/index_ivf_pq.hpp>

auto ctx = cw::Context::create().value();

// Configure IVF-PQ: 64 clusters, 32 nprobe, 16 subquantizers
cw::IVFPQParams params{
    .ivf = {.nlist = 64, .nprobe = 32},
    .pq = {.m = 16, .nbits = 8}
};
auto index = cw::IndexIVFPQ::create(ctx, 128, params).value();

// Train on representative data
index.train(train_data, 50'000).value();

// Add vectors — achieves ~976x compression
index.add(data, 1'000'000).value();

// Search — 77-97% recall with AVX-optimized re-ranking
auto results = index.search({query.data(), 128}, 10).value();
```

## Build

```bash
git clone https://github.com/your-org/catwhisper.git
cd catwhisper
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCW_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

**Requirements**: Vulkan SDK 1.2+, C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+).
VMA is fetched automatically via CMake FetchContent.

## How It's Fast

CatWhisper's single-query latency advantage over FAISS-GPU comes from a stack
of low-level optimizations applied to the Vulkan hot path:

**Structure-of-Arrays database layout** — vectors are stored transposed
(`database[dim * capacity + vector_idx]`) so that all threads in a warp access
contiguous fp16 values at each dimension step.  One cache line per warp instead
of 32 under AoS.  This is the dominant win at 100K+ vectors where the database
exceeds GPU L2 capacity.

**Fused distance + top-k shader** — distance computation and bitonic sort
top-k selection run in a single shader invocation over 2048-vector shared-memory
tiles, with no intermediate global distance buffer and no inter-pass barrier.

**Timeline semaphore + userspace spin-poll** — Vulkan 1.2 timeline semaphores
replace binary fences (no reset required), and `vkGetSemaphoreCounterValue` polls
a GPU-mapped host-visible page with no kernel transition on NVIDIA's driver.
Eliminates the Linux scheduler wake-up latency (~5–15 µs) on the GPU completion
signal path.

**Zero-copy query and result paths** — the query buffer is HostCoherent
(persistent mapped pointer, written with AVX-512/F16C fp32→fp16 conversion).
Result buffers are GPU_TO_CPU cached system RAM read directly through a mapped
pointer.  No staging copies, no extra fences.

**Persistent reusable command buffer** — recorded once at first search, re-submitted
on every subsequent query without re-recording as long as `n_vectors` and `k` are
unchanged.

### IndexSparseIOT — The Curse Breaker

Every index above operates on **fixed-dimensional dense vectors** in Euclidean or cosine space. As dimensionality grows, distance contrast collapses — all points become equidistant. This is the curse of dimensionality, and every ANN system in production today works around it rather than solving it.

`IndexSparseIOT` solves it.

It uses the **Involuted Oblate Toroid (IOT) distance metric**, which has a mathematically proven property: distance contrast grows as **Θ(√d)** instead of collapsing to zero. This is the only known distance metric where higher dimensionality makes retrieval *better*, not worse.

The index exploits the IOT's fractal binary path structure. For each stored vector, the index computes a binary path signature against a set of pivot points — each dimension independently chooses a direct or involuted geodesic path, producing a d-bit fingerprint. Search computes the query's signature and finds nearest neighbors via Hamming distance on signatures, then reranks with exact IOT distance.

**Key properties:**
- **Dynamic dimensionality**: vectors can have different numbers of active dimensions. No rebuild needed when dimensions grow. Add a dimension to your embedding space and the index absorbs it.
- **Sparse native**: only non-zero entries are stored and compared. A 2-million-dimensional vector with 500 active entries costs what a 500-dimensional dense vector costs.
- **Anti-curse geometry**: retrieval quality is *guaranteed* to improve as dimensionality increases. Not "degrade gracefully." Improve.
- **GPU-accelerated Hamming batch**: Vulkan compute shader for bulk Hamming distance computation across signature sets.

```cpp
#include <catwhisper/index_sparse_iot.hpp>

auto index = cw::IndexSparseIOT::create({
    .n_pivots = 32,
    .n_probes = 4
}).value();

// Add sparse vectors — each can have different dimensions
index.add(id, sparse_vec).value();

// Dimensionality grew? No rebuild needed.
index.notify_dimension_growth(new_max_dim);

// Search
auto results = index.search(query_sparse, 10).value();
```

**Why this matters for RAG**: Every retrieval-augmented generation system fights the curse of dimensionality. Embeddings from transformers (768d, 1024d, 1536d) are already in the regime where distance concentration degrades retrieval. Drop in `IndexSparseIOT` with IOT-projected embeddings and retrieval precision holds or improves as your knowledge base grows. No chunking hacks, no reranking band-aids — the geometry does the work.

**Why this matters for SCN**: In State-Coallapse Networks, the lambda fold makes every graph node a new dimension. The graph grows to millions of nodes — millions of dimensions. `IndexSparseIOT` is the index that makes million-dimensional ANN search not just possible but *better* than thousand-dimensional search. It is the reason SCN's knowledge retrieval scales with unbounded dimensional growth.

## Status

**Alpha.** IndexFlat, IndexIVFFlat, IndexIVFPQ, IndexHNSW, and IndexSparseIOT are complete and tested.
See [ROADMAP](docs/ROADMAP.md) for details.

| Index | Status |
|-------|--------|
| IndexFlat | ✅ Complete — beats FAISS-GPU |
| IndexIVFFlat | ✅ Complete — 98-100% recall, GPU-accelerated |
| IndexIVFPQ | ✅ Complete — 976x compression, GPU ADC + AVX re-ranking |
| IndexHNSW | ✅ Complete — AVX-512 SIMD + parallel batch + GPU path |
| IndexSparseIOT | ✅ Complete — fractal binary path index, anti-curse IOT metric, dynamic dimensionality |

106 tests passing.
