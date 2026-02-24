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

**Alpha.** IndexFlat is complete and production-tested. IndexIVFFlat, IndexIVFPQ,
and IndexHNSW are on the roadmap; see [ROADMAP](docs/ROADMAP.md).

| Index | Status |
|-------|--------|
| IndexFlat | ✅ Complete — beats FAISS-GPU |
| IndexIVFFlat | ❌ Not started |
| IndexIVFPQ | ❌ Not started |
| IndexHNSW | ❌ Not started |

41 unit tests + 3 performance budget tests passing.

## Architecture

- [Architecture Overview](docs/ARCHITECTURE.md)
- [GPU Compute Pipeline](docs/GPU_COMPUTE.md)
- [Data Structures & Algorithms](docs/DATA_STRUCTURES.md)
- [API Design](docs/API_DESIGN.md)
- [Build System](docs/BUILD_SYSTEM.md)
- [Development Roadmap](docs/ROADMAP.md)

## License

MIT or Apache 2.0 (dual-licensed).
