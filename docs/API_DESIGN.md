# API Design

## Design Principles

1. **Safety First**: Use `std::expected` for error handling, no exceptions
2. **RAII**: All resources automatically managed
3. **Modern C++**: C++20 features (concepts, ranges, spans)
4. **Zero-cost Abstractions**: Pay only for what you use
5. **Clear Ownership**: No shared_ptr unless truly shared

## Public Header

```cpp
// include/catwhisper/catwhisper.hpp

#ifndef CATWHISPER_HPP
#define CATWHISPER_HPP

#include <catwhisper/version.hpp>
#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/index.hpp>
#include <catwhisper/distance.hpp>

#endif // CATWHISPER_HPP
```

## Core Types

```cpp
// include/catwhisper/types.hpp

namespace cw {

// Vector data types
using Vector = std::span<const float>;
using VectorMut = std::span<float>;
using VectorId = uint64_t;

// Distance metric enumeration
enum class Metric : uint8_t {
    L2 = 0,      // Euclidean distance
    IP = 1,      // Inner product (negative)
    Cosine = 2   // Cosine distance
};

// Search result
struct SearchResult {
    VectorId id;
    float distance;
};

// Batch of search results
struct SearchResults {
    std::vector<SearchResult> results;  // k results per query
    uint32_t n_queries;
    uint32_t k;
    
    // Access results for query i
    std::span<const SearchResult> operator[](uint32_t i) const {
        return {results.data() + i * k, k};
    }
};

// Index statistics
struct IndexStats {
    uint64_t n_vectors;
    uint32_t dimension;
    uint64_t memory_used;
    uint64_t gpu_memory_used;
    bool is_trained;
};

// GPU device info
struct DeviceInfo {
    uint32_t device_id;
    std::string name;
    uint64_t total_memory;
    uint64_t available_memory;
    std::string driver_version;
    bool supports_fp16;
    bool supports_int8;
};

} // namespace cw
```

## Error Handling

```cpp
// include/catwhisper/error.hpp

namespace cw {

enum class ErrorCode : int {
    Success = 0,
    
    // Initialization errors
    VulkanInitFailed = 100,
    NoComputeCapableDevice = 101,
    DeviceCreationFailed = 102,
    
    // Memory errors
    OutOfGPUMemory = 200,
    OutOfHostMemory = 201,
    BufferCreationFailed = 202,
    
    // Index errors
    IndexNotTrained = 300,
    InvalidDimension = 301,
    InvalidParameter = 302,
    IndexFull = 303,
    
    // IO errors
    FileNotFound = 400,
    InvalidFileFormat = 401,
    WriteFailed = 402,
    ReadFailed = 403,
    
    // Operation errors
    OperationFailed = 500,
    Timeout = 501,
    DeviceLost = 502
};

class Error {
public:
    Error(ErrorCode code, std::string message = "")
        : code_(code), message_(std::move(message)) {}
    
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }
    
    explicit operator bool() const { return code_ != ErrorCode::Success; }
    
    // For compatibility with expected
    friend bool operator==(const Error& lhs, const Error& rhs) {
        return lhs.code_ == rhs.code_;
    }
    
private:
    ErrorCode code_;
    std::string message_;
};

// Convenience aliases
template<typename T>
using Expected = std::expected<T, Error>;

using Unexpected = std::unexpected<Error>;

// Helper macros (internal use)
#define CW_EXPECT(expr) \
    if (auto _result = (expr); !_result) { \
        return std::unexpected(std::move(_result).error()); \
    }

#define CW_EXPECT_VAL(var, expr) \
    auto _##var = (expr); \
    if (!_##var) { \
        return std::unexpected(std::move(_##var).error()); \
    } \
    auto var = std::move(*_##var);

} // namespace cw
```

## Context

```cpp
// include/catwhisper/context.hpp

namespace cw {

struct ContextOptions {
    // Device selection
    int device_id = -1;  // -1 = auto-select best
    
    // Memory limits
    uint64_t max_gpu_memory = 0;  // 0 = use all available
    
    // Debug options
    bool enable_validation = false;
    bool enable_debug_names = false;
    
    // Performance options
    uint32_t num_queues = 1;  // Number of compute queues
};

class Context {
public:
    // Non-copyable, movable
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;
    ~Context();
    
    // Factory function
    [[nodiscard]] static Expected<Context> create(const ContextOptions& options = {});
    
    // Device information
    const DeviceInfo& device_info() const;
    
    // List all available devices
    [[nodiscard]] static Expected<std::vector<DeviceInfo>> list_devices();
    
    // Memory information
    uint64_t total_gpu_memory() const;
    uint64_t available_gpu_memory() const;
    
    // Synchronization
    void synchronize();  // Wait for all GPU operations to complete
    
    // Raw Vulkan access (for advanced users)
    void* vulkan_device();  // Returns VkDevice
    void* vulkan_instance();  // Returns VkInstance
    
private:
    Context() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw
```

## Index Base Class

```cpp
// include/catwhisper/index.hpp

namespace cw {

// Common index options
struct IndexOptions {
    Metric metric = Metric::L2;
    bool use_fp16 = true;  // Use float16 for GPU storage
};

// Abstract index interface
class IndexBase {
public:
    virtual ~IndexBase() = default;
    
    // Properties
    virtual uint32_t dimension() const = 0;
    virtual uint64_t size() const = 0;
    virtual bool is_trained() const = 0;
    virtual IndexStats stats() const = 0;
    
    // Training (if applicable)
    virtual Expected<void> train(std::span<const float> data, uint64_t n) {
        return {};  // Default: no training needed
    }
    
    // Adding vectors
    virtual Expected<void> add(std::span<const float> data, uint64_t n,
                               std::span<const VectorId> ids = {}) = 0;
    
    // Single query search
    virtual Expected<SearchResults> search(Vector query, uint32_t k) = 0;
    
    // Batch search (more efficient)
    virtual Expected<SearchResults> search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) = 0;
    
    // Remove vectors (if supported)
    virtual Expected<void> remove(std::span<const VectorId> ids) {
        return Unexpected(Error(ErrorCode::InvalidParameter, 
                               "Remove not supported for this index type"));
    }
    
    // Serialization
    virtual Expected<void> save(const std::filesystem::path& path) const = 0;
    virtual Expected<void> load(const std::filesystem::path& path) = 0;
    
    // Save/load to/from binary buffer
    virtual Expected<std::vector<uint8_t>> serialize() const = 0;
    virtual Expected<void> deserialize(std::span<const uint8_t> data) = 0;
    
    // Reset the index
    virtual void reset() = 0;
};

} // namespace cw
```

## IndexFlat

```cpp
// include/catwhisper/index_flat.hpp

namespace cw {

class IndexFlat : public IndexBase {
public:
    // Factory
    [[nodiscard]] static Expected<IndexFlat> create(
        Context& ctx,
        uint32_t dimension,
        const IndexOptions& options = {}
    );
    
    // Move-only
    IndexFlat(IndexFlat&&) noexcept;
    IndexFlat& operator=(IndexFlat&&) noexcept;
    ~IndexFlat();
    
    // Implement IndexBase interface
    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override { return true; }  // Always trained
    IndexStats stats() const override;
    
    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;
    
    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;
    
    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;
    Expected<std::vector<uint8_t>> serialize() const override;
    Expected<void> deserialize(std::span<const uint8_t> data) override;
    
    void reset() override;
    
    // IndexFlat-specific: get vector by ID
    Expected<std::vector<float>> get_vector(VectorId id) const;
    
private:
    IndexFlat() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw
```

## IndexIVFFlat

```cpp
// include/catwhisper/index_ivf_flat.hpp

namespace cw {

struct IVFParams {
    uint32_t nlist = 0;     // Number of clusters (0 = auto, sqrt(n))
    uint32_t nprobe = 1;    // Clusters to search (1 = fastest, lowest recall)
    
    // Training options
    uint32_t kmeans_iters = 10;
    uint32_t kmeans_seed = 0;  // 0 = random
    
    // For search: can override nprobe
    IVFParams with_nprobe(uint32_t nprobe_) const {
        IVFParams p = *this;
        p.nprobe = nprobe_;
        return p;
    }
};

class IndexIVFFlat : public IndexBase {
public:
    // Factory
    [[nodiscard]] static Expected<IndexIVFFlat> create(
        Context& ctx,
        uint32_t dimension,
        const IVFParams& ivf_params = {},
        const IndexOptions& options = {}
    );
    
    // Move-only
    IndexIVFFlat(IndexIVFFlat&&) noexcept;
    IndexIVFFlat& operator=(IndexIVFFlat&&) noexcept;
    ~IndexIVFFlat();
    
    // Implement IndexBase interface
    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override;
    IndexStats stats() const override;
    
    Expected<void> train(std::span<const float> data, uint64_t n) override;
    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;
    
    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;
    
    // Search with custom nprobe
    Expected<SearchResults> search(Vector query, uint32_t k, uint32_t nprobe);
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k, uint32_t nprobe);
    
    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;
    Expected<std::vector<uint8_t>> serialize() const override;
    Expected<void> deserialize(std::span<const uint8_t> data) override;
    
    void reset() override;
    
    // IVF-specific operations
    uint32_t nlist() const;
    uint32_t nprobe() const;
    void set_nprobe(uint32_t nprobe);
    
    // Get cluster assignment for a vector
    Expected<uint32_t> assign_cluster(Vector query) const;
    
    // Get vectors in a specific cluster
    Expected<std::vector<VectorId>> get_cluster_vectors(uint32_t cluster) const;
    
private:
    IndexIVFFlat() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw
```

## IndexIVFPQ

```cpp
// include/catwhisper/index_ivf_pq.hpp

namespace cw {

struct PQParams {
    uint32_t m = 0;         // Number of subquantizers (0 = auto, d/16)
    uint32_t nbits = 8;     // Bits per subquantizer (8 = 256 centroids)
};

struct IVFPQParams {
    IVFParams ivf;
    PQParams pq;
};

class IndexIVFPQ : public IndexBase {
public:
    // Factory
    [[nodiscard]] static Expected<IndexIVFPQ> create(
        Context& ctx,
        uint32_t dimension,
        const IVFPQParams& params = {},
        const IndexOptions& options = {}
    );
    
    // Move-only
    IndexIVFPQ(IndexIVFPQ&&) noexcept;
    IndexIVFPQ& operator=(IndexIVFPQ&&) noexcept;
    ~IndexIVFPQ();
    
    // Implement IndexBase interface
    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override;
    IndexStats stats() const override;
    
    Expected<void> train(std::span<const float> data, uint64_t n) override;
    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;
    
    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;
    
    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;
    Expected<std::vector<uint8_t>> serialize() const override;
    Expected<void> deserialize(std::span<const uint8_t> data) override;
    
    void reset() override;
    
    // PQ-specific operations
    uint32_t m() const;       // Number of subquantizers
    uint32_t nbits() const;   // Bits per subquantizer
    uint64_t bytes_per_vector() const;  // m * ceil(nbits/8)
    
private:
    IndexIVFPQ() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw
```

## IndexHNSW

```cpp
// include/catwhisper/index_hnsw.hpp

namespace cw {

struct HNSWParams {
    uint32_t M = 16;                  // Max connections per node
    uint32_t ef_construction = 200;   // Build-time search width
    uint32_t ef_search = 50;          // Search-time width
    double ml = 0.0;                  // Level multiplier (0 = auto, 1/ln(M))
};

class IndexHNSW : public IndexBase {
public:
    // Factory
    [[nodiscard]] static Expected<IndexHNSW> create(
        uint32_t dimension,
        const HNSWParams& params = {},
        const IndexOptions& options = {}
    );
    
    // Note: HNSW is CPU-only, no Context needed
    
    // Move-only
    IndexHNSW(IndexHNSW&&) noexcept;
    IndexHNSW& operator=(IndexHNSW&&) noexcept;
    ~IndexHNSW();
    
    // Implement IndexBase interface
    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override { return true; }  // HNSW doesn't need training
    IndexStats stats() const override;
    
    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;
    
    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;
    
    // Search with custom ef
    Expected<SearchResults> search(Vector query, uint32_t k, uint32_t ef);
    
    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;
    Expected<std::vector<uint8_t>> serialize() const override;
    Expected<void> deserialize(std::span<const uint8_t> data) override;
    
    void reset() override;
    
    // HNSW-specific operations
    uint32_t M() const;
    uint32_t ef_construction() const;
    uint32_t ef_search() const;
    void set_ef_search(uint32_t ef);
    
    // Get connectivity info
    uint32_t max_level() const;
    uint32_t entry_point() const;
    uint32_t node_level(VectorId id) const;
    std::vector<VectorId> neighbors(VectorId id, uint32_t level = 0) const;
    
private:
    IndexHNSW() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw
```

## Distance Functions

```cpp
// include/catwhisper/distance.hpp

namespace cw {

// CPU distance functions
namespace distance {

inline float l2_sqr(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float inner_product(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;  // Positive = more similar
}

inline float cosine_similarity(std::span<const float> a, std::span<const float> b) {
    float dot = inner_product(a, b);
    float norm_a = std::sqrt(l2_sqr(a, std::span<float>{}));
    float norm_b = std::sqrt(l2_sqr(b, std::span<float>{}));
    return dot / (norm_a * norm_b);
}

// Normalization helper
inline void normalize(std::span<float> vec) {
    float norm = std::sqrt(l2_sqr(vec, std::span<float>{}));
    for (auto& v : vec) {
        v /= norm;
    }
}

inline std::vector<float> normalized(std::span<const float> vec) {
    std::vector<float> result(vec.begin(), vec.end());
    normalize(result);
    return result;
}

} // namespace distance

// Batch normalization
inline void normalize_batch(std::span<float> data, uint64_t n, uint32_t dim) {
    for (uint64_t i = 0; i < n; ++i) {
        distance::normalize(data.subspan(i * dim, dim));
    }
}

} // namespace cw
```

## Usage Examples

### Basic Usage

```cpp
#include <catwhisper/catwhisper.hpp>
#include <iostream>

int main() {
    namespace cw = catwhisper;
    
    // Create context
    auto ctx_result = cw::Context::create();
    if (!ctx_result) {
        std::cerr << "Failed to create context: " 
                  << ctx_result.error().message() << "\n";
        return 1;
    }
    auto ctx = std::move(*ctx_result);
    
    std::cout << "Using GPU: " << ctx.device_info().name << "\n";
    
    // Create flat index
    auto index_result = cw::IndexFlat::create(ctx, 128);
    if (!index_result) {
        std::cerr << "Failed to create index\n";
        return 1;
    }
    auto index = std::move(*index_result);
    
    // Add some vectors (in practice, load your data)
    std::vector<float> data(10000 * 128);
    // ... fill data ...
    
    auto add_result = index.add(data, 10000);
    if (!add_result) {
        std::cerr << "Failed to add vectors\n";
        return 1;
    }
    
    // Search
    std::vector<float> query(128);
    // ... fill query ...
    
    auto search_result = index.search(query, 10);
    if (!search_result) {
        std::cerr << "Search failed\n";
        return 1;
    }
    
    for (const auto& result : (*search_result)[0]) {
        std::cout << "ID: " << result.id 
                  << ", Distance: " << result.distance << "\n";
    }
    
    return 0;
}
```

### IVF with Tuning

```cpp
// Create IVF index with parameters
cw::IVFParams ivf_params;
ivf_params.nlist = 256;     // sqrt(65536) = 256 clusters
ivf_params.nprobe = 16;     // Search 16 clusters

auto index = cw::IndexIVFFlat::create(ctx, 128, ivf_params).value();

// Train
index.train(training_data, 100000).value();

// Add
index.add(database, 1000000).value();

// Search with different nprobe for different accuracy/speed tradeoffs
auto fast_results = index.search(query, 10, /*nprobe=*/4);   // Fast, lower recall
auto accurate_results = index.search(query, 10, /*nprobe=*/64); // Slower, higher recall
```

### Batch Search

```cpp
// Batch search is much more efficient on GPU
std::vector<float> queries(1000 * 128);  // 1000 queries
// ... fill queries ...

auto results = index.search(queries, 1000, 10).value();

for (uint64_t q = 0; q < 1000; ++q) {
    auto query_results = results[q];
    // Process results for query q
    for (const auto& r : query_results) {
        // ...
    }
}
```

### Save and Load

```cpp
// Save index
index.save("my_index.cw").value();

// Load index
auto loaded = cw::IndexIVFFlat::create(ctx, 128).value();
loaded.load("my_index.cw").value();

// Or serialize to memory
auto buffer = index.serialize().value();
auto index2 = cw::IndexIVFFlat::create(ctx, 128).value();
index2.deserialize(buffer).value();
```

### Memory-mapped Load

```cpp
// For large indexes, memory-mapped loading is faster
// (implementation detail, may be added later)
auto index = cw::IndexIVFFlat::mmap_load(ctx, "large_index.cw").value();
```
