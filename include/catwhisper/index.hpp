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
    bool use_int8 = false;   // Quantize to int8; 4× storage reduction, faster GPU kernels.
                              // use_int8 takes precedence over use_fp16 when both are set.
                              // Requires Vulkan 1.2+ with shaderInt8 feature.
    float int8_scale = 127.0f; // Quantization scale: quantized = clamp(round(v * int8_scale), -127, 127).
                                // Default 127 assumes values in [-1, 1].
                                // Set to 127 / max_abs_expected for your data distribution.
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
