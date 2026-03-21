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
    float ml_factor = 0.0f;              // Level multiplier (0 = auto: 1/ln(M))
    bool use_bf16_storage = false;        // Store vectors as BF16; halves CPU search bandwidth.
                                          // Requires __AVX512BF16__ for fast conversion path.
                                          // float data is always kept for GPU upload.
    bool use_int8_storage = false;        // Store vectors as symmetric int8; 4× storage reduction.
                                          // use_int8_storage takes precedence over use_bf16_storage.
    float int8_scale = 127.0f;            // Quantization scale: quantized = clamp(round(v * int8_scale), -127, 127).
                                          // Default 127 assumes values in [-1, 1].
    bool use_int16_storage = false;       // Store vectors natively as int16 — zero conversion.
                                          // Takes precedence over use_int8_storage and use_bf16_storage.
                                          // Distance computed via SIMD int16 dot product with
                                          // precomputed int64 norms for L2/Cosine.
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
    Expected<void> add_int16(std::span<const int16_t> data, uint64_t n,
                             std::span<const VectorId> ids = {}) override;

    Expected<SearchResults> search(Vector query, uint32_t k) override;
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;
    Expected<SearchResults> search_int16(VectorI16 query, uint32_t k) override;
    Expected<SearchResults> search_int16(std::span<const int16_t> queries,
                                         uint64_t n_queries, uint32_t k) override;

    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;

    void reset() override;

    void set_ef_search(uint32_t ef) { ef_search_ = ef; }
    uint32_t ef_search() const { return ef_search_; }

    bool valid() const { return impl_ != nullptr; }
    bool gpu_enabled() const;
    bool gpu_int16_enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint32_t ef_search_ = 50;

    Expected<void> add_single(const float* vec, VectorId id);
    void connect_node(uint32_t node_id, uint32_t level);
    void search_single_locked(const float* query, uint32_t k, SearchResult* out);
    void search_single_locked_int16(const int16_t* query, int64_t query_norm_sq,
                                     uint32_t k, SearchResult* out);
    Expected<SearchResults> search_batch_gpu(std::span<const float> queries,
                                             uint64_t n_queries, uint32_t k);
    Expected<SearchResults> search_batch_gpu_int16(std::span<const int16_t> queries,
                                                    uint64_t n_queries, uint32_t k);
};

} // namespace cw

#endif // CATWHISPER_INDEX_HNSW_HPP
