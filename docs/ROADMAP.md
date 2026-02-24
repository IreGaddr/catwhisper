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
| distance_ip.comp | Inner product shader | Critical | ❌ Not started |
| topk_select.comp | Top-k heap selection | Critical | ⚠️ Partial (CPU sort used) |
| IndexFlat class | Host-side implementation | Critical | ✅ Done |
| Data upload | Host → GPU staging and transfer | Critical | ✅ Done |
| Result download | GPU → Host readback | Critical | ✅ Done |
| Unit tests | IndexFlat test coverage | High | ✅ Done (35 tests) |
| Benchmarks | vs FAISS comparison | Medium | ❌ Not started |

#### Shaders

```
shaders/
├── distance_l2.comp       # L2 distance computation ✅
├── distance_ip.comp       # Inner product computation ❌
├── distance_cosine.comp   # Cosine distance (optional, can normalize + IP) ❌
└── topk_heap.comp         # Heap-based top-k selection ⚠️ (exists, CPU fallback used)
```

#### Deliverables

- [x] IndexFlat::add() works
- [x] IndexFlat::search() returns correct results
- [x] Batch search implemented (loop over single queries)
- [ ] Performance within 80% of FAISS IndexFlat (not benchmarked)

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
Phase 1: IndexFlat      [████████░░] 80%  ✅ Functional (needs GPU top-k)
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
| Top-K Heap | `topk_heap.comp` | ⚠️ Exists but CPU fallback used |

### Tests

All 35 unit tests passing:
- BufferTest (5 tests)
- ContextTest (5 tests)
- ErrorTest (6 tests)
- IndexFlatTest (9 tests)
- MinimalTest (5 tests)
- TypesTest (5 tests)

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| IndexFlat throughput | 10M vectors/sec | ⚠️ Not benchmarked |
| IndexIVFFlat recall@10 | >95% @ nprobe=32 | ❌ Not implemented |
| IndexIVFPQ compression | >20x | ❌ Not implemented |
| IndexHNSW latency | <1ms @ 1M | ❌ Not implemented |
| Test coverage | >80% | ✅ Core functionality tested |
| Build time | <5 min | ✅ Fast |
| GitHub stars | 1000+ | - |

---

*This roadmap is a living document. Expect updates as we learn and grow.*
