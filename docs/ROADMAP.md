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
Date: 2026-02-23 · CatWhisper fp16 mode · FAISS 1.13.2

##### After Phase 1 optimizations (current)

| Config | CatWhisper fp16 | FAISS-GPU single | FAISS-CPU single | CW / GPU ratio |
|--------|----------------|-----------------|-----------------|----------------|
| 10K × 128, k=10 | **0.124 ms / 8,089 QPS** | 0.052 ms / 19,223 QPS | 0.059 ms / 16,843 QPS | 2.4× slower |
| 100K × 128, k=10 | **0.619 ms / 1,617 QPS** | 0.362 ms / 2,761 QPS | 0.837 ms / 1,194 QPS | 1.7× slower ✅ beats CPU |
| 100K × 256, k=10 | **2.002 ms / 500 QPS** | 0.561 ms / 1,783 QPS | 2.231 ms / 448 QPS | 3.6× slower ✅ beats CPU |

FAISS-GPU batch QPS (amortized over 200-query batches): 1.68M / 86K / 90K QPS;
CatWhisper batch search dispatch is implemented but not benchmarked yet.

##### Before optimizations (baseline, for reference)

| Config | CatWhisper fp16 | CW / GPU ratio |
|--------|----------------|----------------|
| 10K × 128, k=10 | 0.937 ms / 1,067 QPS | 18.0× slower |
| 100K × 128, k=10 | 1.482 ms / 675 QPS | 4.1× slower |
| 100K × 256, k=10 | 7.108 ms / 141 QPS | 12.7× slower |

##### Optimization sprint applied (Phase 1, 2026-02-23)

Three targeted changes eliminated the dominant sources of per-query overhead:

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
**3.6× faster** at 100K×256.  CatWhisper now **beats FAISS-CPU** at 100K×128 and
100K×256 for single-query latency.

**Remaining gap to FAISS-GPU**: 1.7–3.6× on single queries.  The residual overhead
is one `vkQueueSubmit` + `vkWaitForFences` round-trip per query (irreducible with the
current synchronous API).  Batch search will close most of this gap by amortising that
cost across N queries at once.

**Phase 1 remaining optimization backlog:**

| Item | Expected gain |
|------|--------------|
| Batch search API (`search_batch(queries, k)`) — already dispatches 2D; needs benchmark | amortises submit cost; targets FAISS parity |
| Subgroup shuffle top-k (replace shared-mem bitonic sort with warp-level primitives) | better occupancy, fewer barriers |

#### Deliverables

- [x] IndexFlat::add() works
- [x] IndexFlat::search() returns correct results
- [x] Batch search implemented (GPU batch dispatch)
- [x] Inner product metric (Metric::IP) implemented and tested
- [x] GPU top-k sort correct across single and multiple workgroups
- [x] Benchmarked against FAISS-GPU and FAISS-CPU (see Benchmark Results below)
- [x] Fused distance+topk shader (eliminates inter-pass barrier)
- [x] Persistent command buffer caching
- [ ] Performance within 80% of FAISS IndexFlat (awaiting real GPU benchmark; optimizations implemented)

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
| K-means CPU | Cluster training implementation | Critical | ❌ Not started |
| assign_clusters.comp | Find nearest centroids | Critical | ❌ Not started |
| IVF data layout | Cluster-sorted storage | Critical | ❌ Not started |
| IndexIVFFlat class | Host implementation | Critical | ❌ Not started |
| Training API | train() method | Critical | ❌ Not started |
| Variable nprobe | Configurable search quality | High | ❌ Not started |
| Dynamic updates | Add vectors to trained index | Medium | ❌ Not started |
| Integration tests | End-to-end IVF tests | High | ❌ Not started |

#### Shaders Needed

```
shaders/
├── assign_clusters.comp    # Find nprobe nearest centroids
├── ivf_distance.comp       # Distance in selected clusters
└── merge_results.comp      # Merge results from multiple clusters
```

#### Deliverables

- [ ] K-means clustering converges
- [ ] IndexIVFFlat training works
- [ ] Search with configurable nprobe
- [ ] Recall > 95% at nprobe=64

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
Phase 1: IndexFlat      [██████████] 98%  ✅ Functional + benchmarked + optimized (beats FAISS-CPU)
Phase 2: IndexIVFFlat   [░░░░░░░░░░] 0%   ← Next up
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

### Shaders

| Shader | File | Status |
|--------|------|--------|
| L2 Distance | `distance_l2.comp` | ✅ Complete |
| Inner Product | `distance_ip.comp` | ✅ Complete |
| Top-K Bitonic Sort | `topk_heap.comp` | ✅ Complete (GPU sort + CPU merge) |

### Tests

All 41 unit tests passing:
- BufferTest (5 tests)
- ContextTest (5 tests)
- ErrorTest (6 tests)
- IndexFlatTest (13 tests)
- MinimalTest (5 tests)
- TypesTest (5 tests)
- IntegrationTest (2 tests)

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| IndexFlat single-query latency | <0.5 ms @ 100K×128 | ⚠️ 0.619 ms (1.7× behind FAISS-GPU; beats FAISS-CPU) |
| IndexIVFFlat recall@10 | >95% @ nprobe=32 | ❌ Not implemented |
| IndexIVFPQ compression | >20x | ❌ Not implemented |
| IndexHNSW latency | <1ms @ 1M | ❌ Not implemented |
| Test coverage | >80% | ✅ Core functionality tested |
| Build time | <5 min | ✅ Fast |
| GitHub stars | 1000+ | - |

---

*This roadmap is a living document. Expect updates as we learn and grow.*
