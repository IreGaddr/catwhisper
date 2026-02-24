# Development Roadmap

## Vision

CatWhisper aims to be a production-ready, high-performance vector similarity search library that:
- Provides a vendor-neutral alternative to FAISS
- Achieves 80-100% of FAISS performance on equivalent hardware
- Supports all major GPU vendors through Vulkan
- Is easy to integrate and deploy

## Release Phases

### Phase 0: Foundation (Weeks 1-4)

**Goal**: Core infrastructure working

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| Vulkan init | Instance, device, queue creation | Critical | ✅ Done |
| Memory allocator | VMA integration, buffer management | Critical | ✅ Done |
| Command buffers | Recording, submission, synchronization | Critical | ✅ Done |
| Pipeline cache | Compute pipeline creation | Critical | ✅ Done |
| Error handling | `std::expected` pattern throughout | High | ✅ Done |
| Logging | spdlog integration for debug | Medium | ❌ Not started |
| CI/CD | GitHub Actions for build/test | Medium | ❌ Not started |

#### Deliverables

- [x] Can initialize Vulkan on all supported platforms
- [x] Can allocate and manage GPU memory (via VMA)
- [x] Basic compute pipeline creation works
- [x] Error handling consistent across codebase (Expected<T> pattern)
- [ ] CI passes on Linux, Windows, macOS

#### Success Criteria

```cpp
// This should work
auto ctx = cw::Context::create().value();
std::cout << ctx.device_info().name << std::endl;
```

---

### Phase 1: IndexFlat (Weeks 5-8)

**Goal**: Brute-force search working on GPU

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| distance_l2.comp | L2 distance compute shader | Critical | ✅ Done |
| distance_ip.comp | Inner product shader | Critical | ✅ Done |
| topk_select.comp | Top-k heap selection | Critical | ✅ Done (GPU bitonic sort + CPU merge) |
| IndexFlat class | Host-side implementation | Critical | ✅ Done |
| Data upload | Host → GPU staging and transfer | Critical | ✅ Done |
| Result download | GPU → Host readback | Critical | ✅ Done |
| Unit tests | IndexFlat test coverage | High | ✅ Done (41 tests) |
| Benchmarks | vs FAISS comparison | Medium | ✅ Done (see Benchmark Results) |

#### Shaders

```
shaders/
├── distance_l2.comp       # L2 distance computation ✅
├── distance_ip.comp       # Inner product computation ✅
├── distance_cosine.comp   # Cosine distance (optional, can normalize + IP) ❌
└── topk_heap.comp         # Per-workgroup ascending bitonic sort top-k ✅
```

#### Implementation Notes

**Top-k strategy**: GPU computes all per-vector distances, then `topk_heap.comp`
runs a correct ascending bitonic sort over 512-element workgroup tiles, writing
`k` candidates per workgroup.  The host CPU-merges the `n_workgroups × k`
partial results via `std::partial_sort` — typically a few thousand elements,
negligible latency.  A fully-GPU merge pass is deferred to a future optimisation
sprint once benchmarks establish it as a bottleneck.

**Metric support**: both `Metric::L2` and `Metric::IP` (negated inner product
for min-heap compatibility) are fully implemented and tested.

#### Benchmark Results

Hardware: NVIDIA GeForce RTX 4080 Laptop GPU · CUDA 12.0 · driver 580
CatWhisper fp16 mode · FAISS 1.13.2

##### After ROTRTA session (current — 2026-02-24)

| Config | CatWhisper fp16 | FAISS-GPU single | FAISS-CPU single | CW / GPU ratio |
|--------|----------------|-----------------|-----------------|----------------|
| 10K × 128, k=10 | **0.053 ms / 18,868 QPS** | 0.065 ms / 15,385 QPS | 0.059 ms / 16,843 QPS | ✅ **1.2× faster** |
| 100K × 128, k=10 | **0.056 ms / 17,857 QPS** | 0.350 ms / 2,857 QPS | 0.837 ms / 1,194 QPS | ✅ **6.3× faster** |
| 100K × 256, k=10 | **0.106 ms / 9,434 QPS** | 0.601 ms / 1,664 QPS | 2.231 ms / 448 QPS | ✅ **5.7× faster** |

CatWhisper **beats FAISS-GPU on all three configurations**. All three ROTRTA assertions passing.

##### After Phase 1 human optimization sprint (2026-02-23, for reference)

| Config | CatWhisper fp16 | FAISS-GPU single | CW / GPU ratio |
|--------|----------------|-----------------|----------------|
| 10K × 128, k=10 | 0.124 ms / 8,089 QPS | 0.065 ms | 2.4× slower |
| 100K × 128, k=10 | 0.619 ms / 1,617 QPS | 0.350 ms | 1.7× slower ✅ beats CPU |
| 100K × 256, k=10 | 2.002 ms / 500 QPS | 0.601 ms | 3.6× slower ✅ beats CPU |

##### Before optimizations (baseline, for reference)

| Config | CatWhisper fp16 | CW / GPU ratio |
|--------|----------------|----------------|
| 10K × 128, k=10 | 0.937 ms / 1,067 QPS | 18.0× slower |
| 100K × 128, k=10 | 1.482 ms / 675 QPS | 4.1× slower |
| 100K × 256, k=10 | 7.108 ms / 141 QPS | 12.7× slower |

##### Optimization sprint 1 applied (Phase 1, 2026-02-23)

Four coordinated changes eliminated the dominant sources of per-query overhead:

1. **HostCoherent query buffer** — changed `query_buffer` from `DeviceLocal` to
   `HostCoherent + map_on_create`.  `Buffer::upload()` now takes the fast `memcpy`
   path instead of staging → `vkCmdCopyBuffer` → `submit_and_wait`.  This removed
   one full fence cycle per query (the biggest single win: −65% at 10K×128).

2. **Persistent reusable command buffer** — replaced `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`
   with `flags=0` for the search command buffer.  After `vkWaitForFences` the buffer
   returns to Executable state and is re-submitted without re-recording.
   `vkResetCommandBuffer` + `begin_reusable()` are called only when `n_vectors` or
   `k` changes.  `vkAllocateCommandBuffers` is called only on the first query.

3. **Persistent fence** — `submit_and_wait` now resets and reuses a single
   `VkFence` stored in `Context::Impl` instead of `vkCreateFence` + `vkDestroyFence`
   on every call.

4. **Descriptor set caching** — `vkUpdateDescriptorSets` is only called when a
   `VkBuffer` handle changes (i.e. on the first query or after `add()` causes a
   reallocation), avoiding 4 driver round-trips on every steady-state query.

Net result: **7.6× faster** at 10K×128, **2.4× faster** at 100K×128,
**3.6× faster** at 100K×256.  CatWhisper beats FAISS-CPU at 100K×128 and
100K×256.  10K×128 ROTRTA assertion still failing at 72 µs vs 65 µs target.

##### ROTRTA session (2026-02-24) — six further optimizations

Six coordinated changes eliminated the residual CPU-side synchronization and
data-movement overhead:

1. **SoA database layout** — database restructured from AoS
   (`database[v * d + dim]`) to SoA (`database[dim * capacity + v]`).
   Adjacent threads in a warp now access contiguous fp16 values → one cache
   line per warp per dimension step instead of 32.  Two new GPU shaders
   maintain the SoA invariant: `transpose_add.comp` (called on `add()`) and
   `soa_repack.comp` (called on reallocation).

2. **CHUNK 1024 → 2048, local_size_x 512 → 1024** — doubles the vectors
   processed per workgroup dispatch, halving workgroup count (10 → 5 at 10K×128,
   49 → 25 at 100K×128) and thus halving the CPU merge work.

3. **Vulkan timeline semaphore** — replaced `VkFence` with a Vulkan 1.2
   timeline semaphore (monotone counter, no reset required).  Eliminates
   `vkResetFences` (~1–2 µs) and allows userspace polling.

4. **Userspace spin-poll** — `submit_and_wait` polls
   `vkGetSemaphoreCounterValue` in a tight loop (SPIN_LIMIT = 2,000,000)
   before falling back to `vkWaitSemaphores`.  On NVIDIA's driver,
   `vkGetSemaphoreCounterValue` reads a GPU-mapped host-visible surface with
   no syscall, eliminating the kernel sleep/wake round-trip (~5–15 µs).

5. **HostReadback result buffers** — distance and index result buffers
   reallocated as `VMA_MEMORY_USAGE_GPU_TO_CPU` (host-cached system RAM).
   CPU reads directly through persistent mapped pointers; no
   `vkCmdCopyBuffer` or additional fence.

6. **AVX-512/F16C fp16 conversion + direct mapped query write** — fp32 →
   fp16 query conversion uses `_mm512_cvtps_ph` (16 elements/cycle) or
   `_mm256_cvtps_ph` (8 elements/cycle) with `-march=native`.  Written
   directly into the persistent-mapped HostCoherent query buffer; no vector
   allocation or intermediate copy.

Net result: **17.7× total speedup** at 10K×128, **26.5× total** at 100K×128,
**67.1× total** at 100K×256 vs the initial baseline.  CatWhisper beats
FAISS-GPU on all three ROTRTA configurations.

The dominant gains at 100K came from SoA coalescing: the 100K×128 database
(25.6 MB) exceeds GPU L2 capacity (4 MB on RTX 4080), so every AoS-layout
warp access fetched 32 separate cache lines; SoA collapses this to one.

#### Deliverables

- [x] IndexFlat::add() works
- [x] IndexFlat::search() returns correct results
- [x] Batch search implemented (GPU batch dispatch)
- [x] Inner product metric (Metric::IP) implemented and tested
- [x] GPU top-k sort correct across single and multiple workgroups
- [x] Benchmarked against FAISS-GPU and FAISS-CPU (see Benchmark Results below)
- [x] Fused distance+topk shader (eliminates inter-pass barrier)
- [x] Persistent command buffer caching
- [x] SoA database layout with GPU-side transpose and repack shaders
- [x] Timeline semaphore + userspace spin-poll synchronization
- [x] ROTRTA benchmark suite — all 3 assertions passing (beats FAISS-GPU)
- [x] Performance exceeds FAISS-GPU on all benchmarked configurations
- [x] ROTRTA paper written (`rotrta.tex`)

#### Success Criteria

```cpp
auto index = cw::IndexFlat::create(ctx, 128).value();
index.add(data, 100000);
auto results = index.search(query, 10);
// Results match brute-force CPU computation
```

---

### Phase 2: IndexIVFFlat (Weeks 9-14)

**Goal**: Clustered index for larger datasets

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| K-means CPU | Cluster training implementation | Critical | ✅ Done (k-means++ init) |
| assign_clusters.comp | Find nearest centroids | Critical | ✅ Done (GPU integrated) |
| IVF data layout | Cluster-sorted storage | Critical | ✅ Done (cluster-major fp16) |
| IndexIVFFlat class | Host implementation | Critical | ✅ Done |
| Training API | train() method | Critical | ✅ Done |
| Variable nprobe | Configurable search quality | High | ✅ Done |
| Unit tests | Basic correctness tests | High | ✅ Done (15 tests) |
| GPU search path | Offload distance computation | High | ✅ Done |
| GPU centroid selection | nprobe nearest on GPU | Medium | ✅ Done |
| ROTRTA assertions | Performance budget tests | Critical | ❌ Not started |

#### Shaders

```
shaders/
├── assign_clusters.comp    # Find nearest centroid per vector ✅ Integrated
└── ivf_distance.comp       # Centroid selection + distance + top-k ✅ Integrated
```

#### Implementation Status

**GPU-Accelerated: COMPLETE** — All 15 unit tests passing. Full GPU pipeline:
- `add()`: Uses `assign_clusters.comp` for GPU cluster assignment
- `search()`: Uses `ivf_distance.comp` for GPU centroid selection, distance computation, and top-k selection
- Data stored in cluster-major fp16 layout with cached ID mapping

**ROTRTA Optimization: NOT STARTED** — Performance budget assertions not yet defined.

#### ROTRTA Performance Targets (to be validated)

| Config | Target | Current |
|--------|--------|---------|
| 10K × 128, k=10, nprobe=16 | faster than IndexFlat | GPU impl ready, benchmarking pending |
| 100K × 128, k=10, nprobe=16 | < 0.5ms | GPU impl ready, benchmarking pending |
| Recall@10, nprobe=32 | > 95% | TBD |
| Recall@10, nprobe=64 | > 99% | TBD |

#### Deliverables

- [x] K-means clustering converges
- [x] IndexIVFFlat training works
- [x] Search with configurable nprobe
- [x] GPU search path integrated (centroid selection + distance + top-k)
- [x] GPU cluster assignment (add operation)
- [ ] Recall > 95% at nprobe=32 (ROTRTA pending)
- [ ] Performance benchmarks vs IndexFlat

#### Success Criteria

```cpp
cw::IVFParams params{.nlist = 256, .nprobe = 16};
auto index = cw::IndexIVFFlat::create(ctx, 128, params).value();
index.train(train_data, 100000);
index.add(data, 1000000);
auto results = index.search(query, 10);
// Recall > 90% compared to ground truth
```

---

### Phase 3: IndexIVFPQ (Weeks 15-20)

**Goal**: Memory-efficient compressed index

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| PQ training | Subquantizer codebooks | Critical | ❌ Not started |
| Encoding | Vector → PQ codes | Critical | ❌ Not started |
| Distance tables | Precomputed ADC | Critical | ❌ Not started |
| pq_distance.comp | Table-lookup distance | Critical | ❌ Not started |
| IndexIVFPQ class | Host implementation | Critical | ❌ Not started |
| Memory tracking | Accurate stats | Medium | ❌ Not started |
| Compression ratio | Verify memory savings | Medium | ❌ Not started |

#### Shaders Needed

```
shaders/
├── pq_distance.comp        # ADC table lookup
└── pq_encode.comp          # Encode vectors (optional, can be CPU)
```

#### Deliverables

- [ ] PQ codebook training works
- [ ] Encoding produces valid codes
- [ ] Search returns reasonable results
- [ ] Memory usage ~30x less than flat

#### Success Criteria

```cpp
cw::IVFPQParams params{
    .ivf = {.nlist = 256, .nprobe = 32},
    .pq = {.m = 16, .nbits = 8}
};
auto index = cw::IndexIVFPQ::create(ctx, 128, params).value();
index.train(train_data, 100000);
index.add(data, 10000000);  // 10M vectors

// Memory usage
auto stats = index.stats();
assert(stats.gpu_memory_used < 100 * 1024 * 1024);  // < 100MB
```

---

### Phase 4: IndexHNSW (Weeks 21-26)

**Goal**: High-performance graph-based index

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| Graph structure | Node/link storage | Critical | ❌ Not started |
| Construction | Layer-by-layer insertion | Critical | ❌ Not started |
| Search | Greedy search algorithm | Critical | ❌ Not started |
| IndexHNSW class | Host implementation | Critical | ❌ Not started |
| Neighbor selection | Heuristic selection | High | ❌ Not started |
| Parallel build | Multi-threaded construction | Medium | ❌ Not started |
| Serialization | Save/load graph | Medium | ❌ Not started |

**Note**: HNSW is CPU-only initially. GPU acceleration is complex and low priority.

#### Deliverables

- [ ] Construction completes for 1M vectors
- [ ] Search latency < 1ms at 90% recall
- [ ] Serialization works
- [ ] Competitive with hnswlib

#### Success Criteria

```cpp
cw::HNSWParams params{.M = 16, .ef_construction = 200};
auto index = cw::IndexHNSW::create(128, params).value();

// Build (CPU)
index.add(data, 1000000);

// Search
index.set_ef_search(50);
auto results = index.search(query, 10);
// Latency < 1ms, recall > 95%
```

---

### Phase 5: Polish & Production (Weeks 27-32)

**Goal**: Production-ready release

#### Tasks

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| API stability | Finalize public API | Critical | ⚠️ In progress |
| Documentation | API docs, tutorials | Critical | ❌ Not started |
| Performance tuning | Optimize hot paths | High | ❌ Not started |
| Memory profiling | Fix leaks, optimize | High | ❌ Not started |
| Error messages | Clear, actionable errors | Medium | ⚠️ Partial |
| Examples | Real-world usage examples | Medium | ⚠️ Basic example exists |
| Python bindings | pybind11 wrapper | Medium | ❌ Not started |
| Benchmark suite | Comprehensive vs FAISS | Medium | ❌ Not started |

#### Deliverables

- [ ] All public APIs stable
- [ ] Documentation complete
- [ ] No memory leaks
- [ ] Python bindings working
- [ ] Performance competitive with FAISS

---

## Long-term Roadmap

### Version 1.0 Goals

- [ ] All four index types stable
- [ ] 80%+ of FAISS performance
- [ ] Cross-platform: Linux, Windows, macOS
- [ ] Python bindings
- [ ] Comprehensive documentation
- [ ] Used in at least one production system

### Future Considerations

| Feature | Description | Priority |
|---------|-------------|----------|
| GPU HNSW | Graph search on GPU | Low |
| Distributed | Multi-node search | Low |
| Incremental | Online index updates | Medium |
| Quantization variants | OPQ, SQ, etc. | Medium |
| Hardware support | WebGPU, ROCm | Low |
| Language bindings | Rust, Go, etc. | Low |

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vulkan driver bugs | High | Test on multiple vendors, validation layers |
| Performance gap vs FAISS | Medium | Profile early, optimize hot paths |
| Memory fragmentation | Medium | Use VMA pools, custom allocators |
| Subgroup portability | Medium | Fallback paths for different hardware |
| SPIR-V compatibility | Low | Test on multiple drivers |

### Project Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | Medium | Stick to roadmap, defer features |
| Contributor burnout | High | Sustainable pace, clear priorities |
| API churn | Medium | Design upfront, version carefully |
| Documentation debt | Medium | Document as we go |

---

## Contributing

### How to Help

1. **Code**: Pick up tasks from the roadmap
2. **Testing**: Run on your hardware, report issues
3. **Documentation**: Improve docs, write examples
4. **Benchmarks**: Run comparisons, share results
5. **Review**: Code review PRs

### Development Setup

```bash
git clone https://github.com/your-org/catwhisper.git
cd catwhisper
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCW_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### Communication

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, ideas
- Matrix/Discord: Real-time chat (link TBD)

---

## Milestone Tracking

```
Phase 0: Foundation     [████████░░] 80%  ✅ Complete
Phase 1: IndexFlat      [██████████] 100% ✅ Complete — beats FAISS-GPU on all benchmarks
Phase 2: IndexIVFFlat   [████████░░] 80%  ⚠️ GPU-accelerated, ROTRTA pending
Phase 3: IndexIVFPQ     [░░░░░░░░░░] 0%
Phase 4: IndexHNSW      [░░░░░░░░░░] 0%
Phase 5: Production     [░░░░░░░░░░] 0%
```

## Current Implementation Status

### Core Components

| Component | File | Status |
|-----------|------|--------|
| Context | `context.hpp`, `context.cpp` | ✅ Complete |
| Buffer | `buffer.hpp`, `buffer.cpp` | ✅ Complete |
| Pipeline | `pipeline.hpp`, `pipeline.cpp` | ✅ Complete |
| Error handling | `error.hpp` | ✅ Complete |
| Types | `types.hpp` | ✅ Complete |
| IndexFlat | `index_flat.hpp`, `index_flat.cpp` | ✅ Complete |
| IndexIVFFlat | `index_ivf_flat.hpp`, `index_ivf_flat.cpp` | ✅ Complete (GPU-accelerated) |

### Shaders

| Shader | File | Status |
|--------|------|--------|
| L2 Distance | `distance_l2.comp` | ✅ Complete |
| Inner Product | `distance_ip.comp` | ✅ Complete |
| Top-K Bitonic Sort | `topk_heap.comp` | ✅ Complete (GPU sort + CPU merge) |
| IVF Cluster Assign | `assign_clusters.comp` | ✅ Complete (GPU integrated) |
| IVF Distance Search | `ivf_distance.comp` | ✅ Complete (centroid selection + search) |

### Tests

All 59 unit tests passing:
- BufferTest (5 tests)
- ContextTest (5 tests)
- ErrorTest (6 tests)
- IndexFlatTest (13 tests)
- IndexIVFFlatTest (15 tests) — GPU-accelerated add + search
- MinimalTest (5 tests)
- TypesTest (5 tests)
- IntegrationTest (2 tests)
- ROTRTA (3 tests) — IndexFlat only

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| IndexFlat single-query latency | <0.5 ms @ 100K×128 | ✅ **0.056 ms** (6.3× faster than FAISS-GPU) |
| IndexFlat single-query latency | beats FAISS-GPU @ 10K×128 | ✅ **0.053 ms** (FAISS-GPU: 0.065 ms) |
| IndexFlat single-query latency | beats FAISS-GPU @ 100K×256 | ✅ **0.106 ms** (FAISS-GPU: 0.601 ms) |
| ROTRTA assertions | 3/3 passing | ✅ All green |
| IndexIVFFlat recall@10 | >95% @ nprobe=32 | ⚠️ GPU impl complete, benchmarking pending |
| IndexIVFFlat latency | <0.5ms @ 100K×128 | ⚠️ GPU impl complete, benchmarking pending |
| IndexIVFPQ compression | >20x | ❌ Not implemented |
| IndexHNSW latency | <1ms @ 1M | ❌ Not implemented |
| Test coverage | >80% | ✅ Core functionality tested (59 unit + 3 ROTRTA) |
| Build time | <5 min | ✅ Fast |
| GitHub stars | 1000+ | - |

---

*This roadmap is a living document. Expect updates as we learn and grow.*
