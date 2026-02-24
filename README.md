# CatWhisper

**Cross-vendor GPU vector similarity search — faster than FAISS-GPU**

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

## Status

**Alpha.** IndexFlat, IndexIVFFlat, and IndexIVFPQ are complete and tested.
IndexHNSW is on the roadmap; see [ROADMAP](docs/ROADMAP.md).

| Index | Status |
|-------|--------|
| IndexFlat | ✅ Complete — beats FAISS-GPU |
| IndexIVFFlat | ✅ Complete — 98-100% recall, GPU-accelerated |
| IndexIVFPQ | ✅ Complete — 976x compression, GPU ADC + AVX re-ranking |
| IndexHNSW | ❌ Not started |

77+ unit tests + 11 performance budget tests passing.
