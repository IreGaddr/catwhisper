#ifndef CATWHISPER_INDEX_IVF_PQ_HPP
#define CATWHISPER_INDEX_IVF_PQ_HPP

#include <catwhisper/index.hpp>
#include <catwhisper/context.hpp>

#include <memory>

namespace cw {

class IndexIVFPQ : public IndexBase {
public:
    [[nodiscard]] static Expected<IndexIVFPQ> create(
        Context& ctx,
        uint32_t dimension,
        const IVFPQParams& params,
        const IndexOptions& options = {}
    );

    IndexIVFPQ();
    IndexIVFPQ(IndexIVFPQ&&) noexcept;
    IndexIVFPQ& operator=(IndexIVFPQ&&) noexcept;
    ~IndexIVFPQ();

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

    void reset() override;

    // IVF-specific methods
    uint32_t nlist() const;
    uint32_t nprobe() const;
    void set_nprobe(uint32_t nprobe);

    // PQ-specific methods
    uint32_t pq_m() const;       // Number of subquantizers
    uint32_t pq_nbits() const;   // Bits per code
    uint32_t pq_subdim() const;  // Dimension per subquantizer

    // Re-ranking methods
    uint32_t rerank_factor() const;
    void set_rerank_factor(uint32_t factor);

    bool valid() const { return impl_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Expected<void> init_pipelines();
    Expected<void> upload_to_gpu();
};

} // namespace cw

#endif // CATWHISPER_INDEX_IVF_PQ_HPP
