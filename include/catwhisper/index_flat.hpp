#ifndef CATWHISPER_INDEX_FLAT_HPP
#define CATWHISPER_INDEX_FLAT_HPP

#include <catwhisper/index.hpp>
#include <catwhisper/context.hpp>

#include <memory>

namespace cw {

class IndexFlat : public IndexBase {
public:
    [[nodiscard]] static Expected<IndexFlat> create(
        Context& ctx,
        uint32_t dimension,
        const IndexOptions& options = {}
    );

    IndexFlat() = default;
    IndexFlat(IndexFlat&&) noexcept;
    IndexFlat& operator=(IndexFlat&&) noexcept;
    ~IndexFlat();

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

    Expected<std::vector<float>> get_vector(VectorId id) const;

    bool valid() const { return impl_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    Expected<void> init_pipelines();
    Expected<void> reallocate_buffers(uint64_t new_capacity);
};

} // namespace cw

#endif // CATWHISPER_INDEX_FLAT_HPP
