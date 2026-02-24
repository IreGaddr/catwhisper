#ifndef CATWHISPER_TYPES_HPP
#define CATWHISPER_TYPES_HPP

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cw {

using Vector = std::span<const float>;
using VectorMut = std::span<float>;
using VectorId = uint64_t;

enum class Metric : uint8_t {
    L2 = 0,
    IP = 1,
    Cosine = 2
};

struct SearchResult {
    VectorId id = 0;
    float distance = 0.0f;
};

class SearchResults {
public:
    std::vector<SearchResult> results;
    uint32_t n_queries;
    uint32_t k;

    SearchResults() : n_queries(0), k(0) {}

    SearchResults(uint32_t n_queries, uint32_t k)
        : results(n_queries * k), n_queries(n_queries), k(k) {}

    std::span<const SearchResult> operator[](uint32_t i) const {
        return {results.data() + static_cast<size_t>(i) * k, k};
    }

    std::span<SearchResult> operator[](uint32_t i) {
        return {results.data() + static_cast<size_t>(i) * k, k};
    }
};

struct IndexStats {
    uint64_t n_vectors = 0;
    uint32_t dimension = 0;
    uint64_t memory_used = 0;
    uint64_t gpu_memory_used = 0;
    bool is_trained = false;
};

struct IVFParams {
    uint32_t nlist = 256;    // Number of clusters/Voronoi cells
    uint32_t nprobe = 1;     // Number of clusters to search
    uint32_t kmeans_iters = 20;  // K-means iterations for training
};

struct DeviceInfo {
    uint32_t device_id = 0;
    std::string name;
    uint64_t total_memory = 0;
    uint64_t available_memory = 0;
    std::string driver_version;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    uint32_t subgroup_size = 0;
    uint32_t max_workgroup_size = 0;
};

} // namespace cw

#endif // CATWHISPER_TYPES_HPP
