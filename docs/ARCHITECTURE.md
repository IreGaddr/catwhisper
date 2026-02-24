# Architecture Overview

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (User code performing search, clustering, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      C++ Public API                          │
│  (catwhisper.hpp - Index classes, Context, Results)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Library Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Index   │  │ Distance │  │  Memory  │  │   Thread     │ │
│  │ Factory  │  │ Metrics  │  │  Manager │  │    Pool      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      GPU Abstraction                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Vulkan Compute Engine                    │   │
│  │  ┌────────┐  ┌────────┐  ┌─────────┐  ┌───────────┐  │   │
│  │  │ Shader │  │ Buffer │  │ Command │  │  Memory   │  │   │
│  │  │ Manager│  │  Pool  │  │  Queue  │  │  Allocator│  │   │
│  │  └────────┘  └────────┘  └─────────┘  └───────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Platform / Drivers                        │
│     Vulkan API → GPU Drivers (NVIDIA/AMD/Intel/etc.)         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Context (`cw::Context`)

The Context is the root object that owns GPU resources. It encapsulates:

- Vulkan instance, physical device, logical device
- Command pools and queues
- Memory allocator (VMA or custom)
- Pipeline cache

**Lifetime**: One context per application (or per GPU). Expensive to create, cheap to share.

```cpp
class Context {
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    VmaAllocator allocator_;
    
    // Singleton-ish pattern with explicit creation
    static expected<Context, Error> create(const ContextOptions& opts = {});
};
```

### 2. Index Types

Each index type implements a common interface but has different performance characteristics:

| Index Type | Build Time | Search Speed | Memory | Accuracy |
|------------|------------|--------------|--------|----------|
| IndexFlat  | O(n)       | O(n)         | O(nd)  | 100%     |
| IndexIVFFlat | O(nk)    | O(nk/nlist)  | O(nd)  | ~95-99%  |
| IndexIVFPQ | O(nk)      | O(nk/nlist)  | O(npq) | ~90-98%  |
| IndexHNSW  | O(n log n) | O(log n)     | O(nd)  | ~95-99%  |

All indexes support:
- `train(data, n)` - Learn index parameters
- `add(data, n)` - Add vectors to index
- `search(queries, k)` - Find k nearest neighbors
- `save(path)` / `load(path)` - Serialization

### 3. GPU Memory Manager

Handles the complexity of GPU memory:

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Manager                            │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │   GPU Memory    │  │         Staging Buffers          │   │
│  │   (VRAM)        │  │     (Host-visible for transfer)  │   │
│  │                 │  │                                  │   │
│  │  ┌───────────┐  │  │  ┌────────────────────────────┐ │   │
│  │  │ IndexData │  │  │  │ Upload Buffer (ring)       │ │   │
│  │  ├───────────┤  │  │  ├────────────────────────────┤ │   │
│  │  │ QueryBuf  │  │  │  │ Download Buffer (ring)     │ │   │
│  │  ├───────────┤  │  │  └────────────────────────────┘ │   │
│  │  │ ResultBuf │  │  │                                  │   │
│  │  └───────────┘  │  │                                  │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
│                                                              │
│  For datasets > VRAM: Paged storage with async streaming    │
└─────────────────────────────────────────────────────────────┘
```

### 4. Compute Pipeline Architecture

Each index type has corresponding compute shaders:

```
IndexFlat:
  - distance_compute.comp  (compute L2/IP/cosine distances)
  - topk_select.comp        (heap-based top-k selection)

IndexIVFFlat:
  - assign_clusters.comp    (find nearest centroids)
  - distance_compute.comp   (compute distances in clusters)
  - topk_select.comp        (merge results)

IndexIVFPQ:
  - assign_clusters.comp    (find nearest centroids)
  - pq_distance.comp        (table lookup distances)
  - topk_select.comp        (merge results)

IndexHNSW:
  - hnsw_layer_search.comp  (search each layer)
  - hnsw_neighbor.comp      (neighbor gathering)
```

## Data Flow: Search Operation

```
1. Host prepares query vectors
          │
          ▼
2. Upload queries to GPU (staging buffer → device buffer)
          │
          ▼
3. Dispatch compute shaders
   ┌────────────────────────────────────────────┐
   │  a) Compute distances to all vectors       │
   │  b) Maintain top-k heap in shared memory   │
   │  c) Write final k results                  │
   └────────────────────────────────────────────┘
          │
          ▼
4. Read back results (device buffer → staging buffer → host)
          │
          ▼
5. Return SearchResult struct to caller
```

## Thread Safety Model

- **Context**: Thread-safe. Multiple threads can share a context.
- **Index**: Thread-safe for reads (search). Writes (add/train) require external synchronization.
- **GPU Operations**: Internally synchronized via Vulkan queues.

```cpp
// Safe: Multiple threads searching same index
std::vector<std::thread> threads;
for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&]() {
        auto results = index.search(queries, k);  // OK
    });
}

// NOT Safe: Concurrent writes
// Thread 1: index.add(data1, n1);
// Thread 2: index.add(data2, n2);  // DATA RACE
```

## Error Handling Strategy

Using `std::expected` (C++23) or `tl::expected` for C++20:

```cpp
auto ctx = cw::Context::create();
if (!ctx) {
    std::cerr << "Failed: " << ctx.error().message << "\n";
    return;
}

auto index = cw::IndexIVFFlat::create(*ctx, 128);
if (!index) {
    std::cerr << "Failed: " << index.error().message << "\n";
    return;
}
```

## Directory Structure

```
catwhisper/
├── include/
│   └── catwhisper/
│       ├── catwhisper.hpp       # Main include
│       ├── context.hpp
│       ├── index.hpp
│       ├── error.hpp
│       └── types.hpp
├── src/
│   ├── core/
│   │   ├── context.cpp
│   │   ├── index.cpp
│   │   └── memory_manager.cpp
│   ├── gpu/
│   │   ├── vulkan_context.cpp
│   │   ├── pipeline.cpp
│   │   ├── buffer.cpp
│   │   └── command.cpp
│   ├── indexes/
│   │   ├── index_flat.cpp
│   │   ├── index_ivf_flat.cpp
│   │   ├── index_ivf_pq.cpp
│   │   └── index_hnsw.cpp
│   └── distance/
│       ├── l2.cpp
│       ├── ip.cpp
│       └── cosine.cpp
├── shaders/
│   ├── distance_compute.comp
│   ├── topk_select.comp
│   ├── assign_clusters.comp
│   └── ...
├── tests/
│   ├── unit/
│   └── integration/
├── benchmarks/
├── docs/
├── CMakeLists.txt
└── README.md
```

## Dependencies

### Required
- **Vulkan SDK** (1.3+): GPU compute
- **Vulkan Memory Allocator** (VMA): GPU memory management
- **C++20 compiler**: GCC 11+, Clang 14+, MSVC 2022+

### Optional
- **Google Test**: Unit testing
- **Google Benchmark**: Performance testing
- **spdlog**: Logging
- **nlohmann/json**: Serialization format

## Performance Targets

Relative to FAISS on same hardware:

| Operation | Target |
|-----------|--------|
| IndexFlat search | 80-100% of FAISS |
| IndexIVFFlat search | 70-90% of FAISS |
| IndexIVFPQ search | 70-90% of FAISS |
| Index build time | 80-100% of FAISS |
| Memory overhead | <110% of FAISS |

We accept some performance trade-off for Vulkan portability, but aim to be competitive.
