#ifndef CATWHISPER_INDEX_HNSW_HPP
#define CATWHISPER_INDEX_HNSW_HPP

#include <catwhisper/index.hpp>
#include <catwhisper/context.hpp>

#include <memory>
#include <random>

namespace cw {

struct HNSWParams {
    uint32_t M = 16;                      // Max connections per node (layer >= 1)
    uint32_t ef_construction = 200;       // Candidate list size during construction
    float ml_factor = 0.0f;               // Level multiplier (0 = auto: 1/ln(M))
};

struct HNSWGPUOptions {
    bool enable = false;                  // Enable GPU acceleration
    bool use_fp16 = true;                 // Use FP16 for GPU vectors
    uint32_t batch_threshold = 1000;      // Use GPU only for batches >= this size
};

class IndexHNSW : public IndexBase {
public:
    [[nodiscard]] static Expected<IndexHNSW> create(
        uint32_t dimension,
        const HNSWParams& params = {},
        const IndexOptions& options = {}
    );
    
    [[nodiscard]] static Expected<IndexHNSW> create_gpu(
        Context& ctx,
        uint32_t dimension,
        const HNSWParams& params = {},
        const IndexOptions& options = {},
        const HNSWGPUOptions& gpu_options = {}
    );

    IndexHNSW() = default;
    IndexHNSW(IndexHNSW&&) noexcept;
    IndexHNSW& operator=(IndexHNSW&&) noexcept;
    ~IndexHNSW();

    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override { return true; }
    IndexStats stats() const override;

    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;

    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;

    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;

    void reset() override;

    void set_ef_search(uint32_t ef) { ef_search_ = ef; }
    uint32_t ef_search() const { return ef_search_; }

    bool valid() const { return impl_ != nullptr; }
    bool gpu_enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint32_t ef_search_ = 50;

    Expected<void> add_single(const float* vec, VectorId id);
    void search_single_locked(const float* query, uint32_t k, SearchResult* out);
    Expected<SearchResults> search_batch_gpu(std::span<const float> queries,
                                             uint64_t n_queries, uint32_t k);
};

} // namespace cw

#endif // CATWHISPER_INDEX_HNSW_HPP
