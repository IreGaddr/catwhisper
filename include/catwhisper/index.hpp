#ifndef CATWHISPER_INDEX_HPP
#define CATWHISPER_INDEX_HPP

#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>
#include <catwhisper/context.hpp>

#include <filesystem>
#include <memory>
#include <span>

namespace cw {

struct IndexOptions {
    Metric metric = Metric::L2;
    bool use_fp16 = true;
};

class IndexBase {
public:
    virtual ~IndexBase() = default;

    virtual uint32_t dimension() const = 0;
    virtual uint64_t size() const = 0;
    virtual bool is_trained() const = 0;
    virtual IndexStats stats() const = 0;

    virtual Expected<void> train(std::span<const float> data, uint64_t n) {
        (void)data; (void)n;
        return {};
    }

    virtual Expected<void> add(std::span<const float> data, uint64_t n,
                               std::span<const VectorId> ids = {}) = 0;

    virtual Expected<SearchResults> search(Vector query, uint32_t k) = 0;
    virtual Expected<SearchResults> search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) = 0;

    virtual Expected<void> save(const std::filesystem::path& path) const = 0;
    virtual Expected<void> load(const std::filesystem::path& path) = 0;

    virtual void reset() = 0;
};

} // namespace cw

#endif // CATWHISPER_INDEX_HPP
