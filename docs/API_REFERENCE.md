# API Reference

## Table of Contents

- [Context](#context)
- [Index Types](#index-types)
  - [IndexFlat](#indexflat)
  - [IndexIVFFlat](#indexivfflat)
  - [IndexIVFPQ](#indexivfpq)
  - [IndexHNSW](#indexhnsw)
- [Types](#types)
- [Error Handling](#error-handling)
- [Distance Functions](#distance-functions)

---

## Context

The `Context` class manages the GPU device and Vulkan resources. All GPU-based indexes require a context.

### Creation

```cpp
#include <catwhisper/catwhisper.hpp>

// Create with default settings (auto-select best GPU)
auto ctx_result = cw::Context::create();
if (!ctx_result) {
    std::cerr << "Error: " << ctx_result.error().message() << "\n";
    return 1;
}
auto ctx = std::move(*ctx_result);
```

### ContextOptions

```cpp
cw::ContextOptions opts;
opts.device_id = -1;              // -1 = auto-select (default)
opts.max_gpu_memory = 0;          // 0 = use all available
opts.enable_validation = false;   // Enable Vulkan validation layers
opts.enable_debug_names = false;  // Add debug names to Vulkan objects
opts.num_queues = 1;              // Number of compute queues

auto ctx = cw::Context::create(opts).value();
```

### Methods

| Method | Description |
|--------|-------------|
| `static Expected<Context> create(options = {})` | Create a GPU context |
| `static Expected<std::vector<DeviceInfo>> list_devices()` | List all available GPUs |
| `const DeviceInfo& device_info() const` | Get device information |
| `uint64_t total_gpu_memory() const` | Total GPU memory in bytes |
| `uint64_t available_gpu_memory() const` | Available GPU memory |
| `void synchronize()` | Wait for all GPU operations to complete |
| `void* vulkan_device()` | Get raw VkDevice (advanced use) |
| `void* vulkan_instance()` | Get raw VkInstance (advanced use) |

### DeviceInfo

```cpp
struct DeviceInfo {
    uint32_t device_id;        // Device index
    std::string name;          // Device name (e.g., "NVIDIA RTX 4080")
    uint64_t total_memory;     // Total VRAM in bytes
    uint64_t available_memory; // Available VRAM
    std::string driver_version;
    bool supports_fp16;        // float16 support
    bool supports_int8;        // int8 support
    uint32_t subgroup_size;    // Subgroup/warp size
    uint32_t max_workgroup_size;
};
```

---

## Index Types

### IndexFlat

Brute-force exact search. All vectors compared against query. **Fastest for small datasets (<1M vectors)**.

#### Performance

| Config | Latency | vs FAISS-GPU |
|--------|---------|--------------|
| 10K × 128, k=10 | 0.053 ms | **1.2× faster** |
| 100K × 128, k=10 | 0.056 ms | **6.3× faster** |
| 100K × 256, k=10 | 0.106 ms | **5.7× faster** |

#### Creation

```cpp
// Basic creation (L2 metric, fp16 storage)
auto index = cw::IndexFlat::create(ctx, 128).value();

// With options
cw::IndexOptions opts;
opts.metric = cw::Metric::L2;   // or cw::Metric::IP for inner product
opts.use_fp16 = true;           // Use fp16 storage (default, recommended)

auto index = cw::IndexFlat::create(ctx, 128, opts).value();
```

#### Methods

| Method | Description |
|--------|-------------|
| `Expected<void> add(data, n, ids = {})` | Add n vectors (d-dimensional floats) |
| `Expected<SearchResults> search(query, k)` | Single query search |
| `Expected<SearchResults> search(queries, n_queries, k)` | Batch search |
| `Expected<void> save(path)` | Save index to file |
| `Expected<void> load(path)` | Load index from file |
| `Expected<std::vector<float>> get_vector(id)` | Retrieve a vector by ID |
| `void reset()` | Clear all vectors |
| `uint32_t dimension() const` | Vector dimension |
| `uint64_t size() const` | Number of vectors |
| `IndexStats stats() const` | Memory and size stats |

#### Example

```cpp
auto ctx = cw::Context::create().value();
auto index = cw::IndexFlat::create(ctx, 128).value();

// Add 100K vectors
std::vector<float> data(100000 * 128);
// ... fill data ...
index.add(data, 100000).value();

// Search
std::vector<float> query(128);
// ... fill query ...
auto results = index.search(query, 10).value();

for (const auto& r : results[0]) {
    std::cout << "ID: " << r.id << ", Distance: " << r.distance << "\n";
}
```

---

### IndexIVFFlat

Inverted File index with clustered search. **Best for 1M-10M vectors with 95%+ recall**.

#### Performance

| Config | Latency | Recall@10 |
|--------|---------|-----------|
| 10K × 128, nprobe=16 | 0.193 ms | 99.0% |
| 100K × 128, nprobe=16 | 0.964 ms | 98.0% |
| 100K × 256, nprobe=16 | 0.346 ms | 100.0% |

#### IVFParams

```cpp
cw::IVFParams params;
params.nlist = 256;       // Number of clusters (cells)
params.nprobe = 16;       // Clusters to search (1 = fastest, nlist = exact)
params.kmeans_iters = 20; // Training iterations
```

**Guidelines:**
- `nlist`: Use `sqrt(n_vectors)` as starting point (e.g., 256 for 65K vectors)
- `nprobe`: Higher = better recall but slower. Start with 16, tune based on recall needs

#### Creation

```cpp
cw::IVFParams params;
params.nlist = 256;
params.nprobe = 16;

auto index = cw::IndexIVFFlat::create(ctx, 128, params).value();
```

#### Methods

All `IndexFlat` methods plus:

| Method | Description |
|--------|-------------|
| `Expected<void> train(data, n)` | Train clusters on representative data |
| `uint32_t nlist() const` | Number of clusters |
| `uint32_t nprobe() const` | Current nprobe setting |
| `void set_nprobe(uint32_t)` | Change nprobe at runtime |

#### Example

```cpp
auto ctx = cw::Context::create().value();

cw::IVFParams params;
params.nlist = 256;
params.nprobe = 16;

auto index = cw::IndexIVFFlat::create(ctx, 128, params).value();

// Train on representative data (10-50K samples)
std::vector<float> train_data(50000 * 128);
// ... fill training data ...
index.train(train_data, 50000).value();

// Add vectors (GPU-accelerated cluster assignment)
std::vector<float> data(1000000 * 128);
index.add(data, 1000000).value();

// Search with adjustable recall/speed tradeoff
index.set_nprobe(8);   // Faster, lower recall
auto fast = index.search(query, 10).value();

index.set_nprobe(64);  // Slower, higher recall
auto accurate = index.search(query, 10).value();
```

---

### IndexIVFPQ

IVF with Product Quantization. **Extreme memory compression (~976x) for 1M+ vectors**.

#### Performance

| Config | Latency | Recall@10 | Memory | Compression |
|--------|---------|-----------|--------|-------------|
| 10K × 128 | 0.58 ms | 77% | 0.02 MB | 488x |
| 100K × 128 | 0.90 ms | 97% | 0.15 MB | 488x |
| 1M × 128 | - | - | 0.25 MB | **976x** |

#### PQParams

```cpp
cw::IVFPQParams params;
params.ivf.nlist = 64;
params.ivf.nprobe = 32;
params.pq.m = 16;        // Number of subquantizers
params.pq.nbits = 8;     // Bits per subquantizer (8 = 256 centroids)
```

**Guidelines:**
- `m`: Dimension must be divisible by m. Common values: 8, 16, 32
- `nbits`: 8 is standard (256 centroids per subquantizer)
- Bytes per vector = `m * ceil(nbits/8)` = 16 bytes for m=16, nbits=8

#### Creation

```cpp
cw::IVFPQParams params;
params.ivf.nlist = 64;
params.ivf.nprobe = 32;
params.pq.m = 16;
params.pq.nbits = 8;

auto index = cw::IndexIVFPQ::create(ctx, 128, params).value();
```

#### Methods

All IVF methods plus:

| Method | Description |
|--------|-------------|
| `uint32_t pq_m() const` | Number of subquantizers |
| `uint32_t pq_nbits() const` | Bits per code |
| `uint32_t pq_subdim() const` | Dimensions per subquantizer |
| `uint32_t rerank_factor() const` | Re-ranking multiplier |
| `void set_rerank_factor(uint32_t)` | Adjust re-ranking (higher = better recall) |

#### Example

```cpp
auto ctx = cw::Context::create().value();

cw::IVFPQParams params;
params.ivf.nlist = 64;
params.ivf.nprobe = 32;
params.pq.m = 16;
params.pq.nbits = 8;

auto index = cw::IndexIVFPQ::create(ctx, 128, params).value();

// Train (needs more data than IVFFlat due to subquantizer codebooks)
std::vector<float> train_data(100000 * 128);
index.train(train_data, 100000).value();

// Add 10M vectors in only ~20 MB GPU memory!
std::vector<float> data(10000000 * 128);
index.add(data, 10000000).value();

auto stats = index.stats();
std::cout << "GPU memory: " << stats.gpu_memory_used / 1024 << " KB\n";
// Output: GPU memory: 19531 KB  (vs 2.4 GB for IndexFlat!)
```

---

### IndexHNSW

Hierarchical Navigable Small World graph. **CPU-only, best recall/speed tradeoff for <10M vectors**.

#### Performance

| Config | Latency | Recall@10 | QPS |
|--------|---------|-----------|-----|
| 10K × 128, M=16, ef=50 | 0.36 ms | 90% | 2829 |
| 100K × 128, M=16, ef=100 | 1.85 ms | 76% | 524 |

#### HNSWParams

```cpp
cw::HNSWParams params;
params.M = 16;                  // Max connections per node
params.ef_construction = 200;   // Build-time candidate list size
params.ml_factor = 0.0f;        // Level multiplier (0 = auto: 1/ln(M))
```

**Guidelines:**
- `M`: Higher = better recall, more memory. Common: 16-48
- `ef_construction`: Higher = better graph quality, slower build. 100-200 typical
- `ef_search`: Set at runtime via `set_ef_search()`. Higher = better recall, slower

#### Creation

```cpp
// Note: No Context needed (CPU-only)
cw::HNSWParams params;
params.M = 16;
params.ef_construction = 200;

auto index = cw::IndexHNSW::create(128, params).value();
index.set_ef_search(50);  // Search-time parameter
```

#### Methods

| Method | Description |
|--------|-------------|
| `Expected<void> add(data, n, ids = {})` | Add vectors (builds graph) |
| `Expected<SearchResults> search(query, k)` | Single query search |
| `Expected<SearchResults> search(queries, n, k)` | Batch search |
| `Expected<void> save(path)` | Save to file |
| `Expected<void> load(path)` | Load from file |
| `void set_ef_search(uint32_t)` | Set search-time ef |
| `uint32_t ef_search() const` | Current ef_search |
| `void reset()` | Clear index |

#### Example

```cpp
cw::HNSWParams params;
params.M = 16;
params.ef_construction = 200;

auto index = cw::IndexHNSW::create(128, params).value();

// Build graph (CPU, can be slow for large datasets)
std::vector<float> data(1000000 * 128);
index.add(data, 1000000).value();

// Fast search with adjustable recall
index.set_ef_search(50);   // Moderate recall, fast
auto results1 = index.search(query, 10).value();

index.set_ef_search(200);  // High recall, slower
auto results2 = index.search(query, 10).value();

// Save/load
index.save("index.hnsw").value();
```

---

## Types

### Vector

```cpp
using Vector = std::span<const float>;
using VectorMut = std::span<float>;
using VectorId = uint64_t;
```

### Metric

```cpp
enum class Metric : uint8_t {
    L2 = 0,      // Euclidean distance (default)
    IP = 1,      // Inner product (negated for min-heap)
    Cosine = 2   // Cosine distance (normalize vectors first)
};
```

### SearchResult / SearchResults

```cpp
struct SearchResult {
    VectorId id;
    float distance;
};

class SearchResults {
    std::vector<SearchResult> results;
    uint32_t n_queries;
    uint32_t k;

    // Access results for query i
    std::span<const SearchResult> operator[](uint32_t i) const;
};
```

### IndexStats

```cpp
struct IndexStats {
    uint64_t n_vectors;
    uint32_t dimension;
    uint64_t memory_used;      // Host memory
    uint64_t gpu_memory_used;  // GPU memory
    bool is_trained;
};
```

---

## Error Handling

CatWhisper uses `std::expected<T, Error>` for error handling (no exceptions).

### Error Class

```cpp
enum class ErrorCode : int {
    Success = 0,

    // Initialization (100-199)
    VulkanInitFailed = 100,
    NoComputeCapableDevice = 101,
    DeviceCreationFailed = 102,

    // Memory (200-299)
    OutOfGPUMemory = 200,
    OutOfHostMemory = 201,
    BufferCreationFailed = 202,

    // Index (300-399)
    IndexNotTrained = 300,
    InvalidDimension = 301,
    InvalidParameter = 302,
    IndexFull = 303,

    // IO (400-499)
    FileNotFound = 400,
    InvalidFileFormat = 401,
    WriteFailed = 402,
    ReadFailed = 403,

    // Operation (500-599)
    OperationFailed = 500,
    Timeout = 501,
    DeviceLost = 502,
    ShaderCompilationFailed = 503,
    PipelineCreationFailed = 504
};

class Error {
    ErrorCode code() const;
    const std::string& message() const;
    explicit operator bool() const;  // true if error
};

template<typename T>
using Expected = std::expected<T, Error>;
```

### Usage Pattern

```cpp
auto result = cw::IndexFlat::create(ctx, 128);
if (!result) {
    const auto& err = result.error();
    switch (err.code()) {
        case cw::ErrorCode::VulkanInitFailed:
            std::cerr << "GPU init failed: " << err.message() << "\n";
            break;
        case cw::ErrorCode::OutOfGPUMemory:
            std::cerr << "Not enough VRAM\n";
            break;
        default:
            std::cerr << "Error: " << err.message() << "\n";
    }
    return 1;
}
auto index = std::move(*result);
```

---

## Distance Functions

CPU distance utilities in `cw::distance` namespace:

```cpp
#include <catwhisper/distance.hpp>

namespace cw::distance {
    // L2 squared distance
    float l2_sqr(std::span<const float> a, std::span<const float> b);

    // Inner product (positive = similar)
    float inner_product(std::span<const float> a, std::span<const float> b);

    // Cosine similarity (-1 to 1)
    float cosine_similarity(std::span<const float> a, std::span<const float> b);

    // Normalize vector in-place
    void normalize(std::span<float> vec);

    // Return normalized copy
    std::vector<float> normalized(std::span<const float> vec);
}

// Batch normalization
void cw::normalize_batch(std::span<float> data, uint64_t n, uint32_t dim);
```

---

## Index Selection Guide

| Dataset Size | Memory Budget | Recall Need | Recommended Index |
|--------------|---------------|-------------|-------------------|
| <100K | Any | 100% | IndexFlat |
| 100K-1M | Plenty | 100% | IndexFlat |
| 100K-10M | Plenty | 95%+ | IndexIVFFlat |
| 1M+ | Limited | 75-95% | IndexIVFPQ |
| <10M | Any | 90%+ | IndexHNSW (CPU) |

### Quick Decision

1. **Need exact results?** → IndexFlat
2. **Limited GPU memory?** → IndexIVFPQ
3. **Best recall/speed on CPU?** → IndexHNSW
4. **Balanced GPU search?** → IndexIVFFlat
