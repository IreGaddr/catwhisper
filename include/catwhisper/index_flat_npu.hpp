#ifndef CATWHISPER_INDEX_FLAT_NPU_HPP
#define CATWHISPER_INDEX_FLAT_NPU_HPP

#include <catwhisper/index.hpp>
#include <catwhisper/npu_backend.hpp>

#include <memory>
#include <mutex>

namespace cw {

struct IndexFlatNpuOptions {
    IndexOptions base;
    uint32_t batch_threshold = 64;   // Min queries to justify NPU dispatch.
                                      // Below this, delegates to CPU distance.
    float int8_scale = 127.0f;       // Quantization scale (must match database).
};

// NPU-accelerated flat index. Batch-only: single-query search falls back to CPU.
// Maintains a persistent int8 copy of all vectors on the NPU (zero-copy on APU).
//
// Intended usage:
//   1. Add vectors via add() (quantizes to int8, uploads to NPU buffer)
//   2. Call sync() after bulk adds to finalize NPU buffer
//   3. Use batch search() for batch queries (routes to NPU when n_queries >= threshold)
//   4. Single search() routes to CPU brute-force
class IndexFlatNpu : public IndexBase {
public:
    [[nodiscard]] static Expected<IndexFlatNpu> create(
        NpuContext& npu_ctx,
        uint32_t dimension,
        const IndexFlatNpuOptions& options = {});

    IndexFlatNpu() = default;
    IndexFlatNpu(IndexFlatNpu&&) noexcept;
    IndexFlatNpu& operator=(IndexFlatNpu&&) noexcept;
    ~IndexFlatNpu();

    uint32_t dimension() const override;
    uint64_t size() const override;
    bool is_trained() const override { return true; }
    IndexStats stats() const override;

    Expected<void> add(std::span<const float> data, uint64_t n,
                       std::span<const VectorId> ids = {}) override;

    // Single query: CPU brute-force fallback (NPU dispatch overhead not worth it).
    Expected<SearchResults> search(Vector query, uint32_t k) override;

    // Batch query: routes to NPU when n_queries >= batch_threshold.
    Expected<SearchResults> search(std::span<const float> queries,
                                   uint64_t n_queries, uint32_t k) override;

    Expected<void> save(const std::filesystem::path& path) const override;
    Expected<void> load(const std::filesystem::path& path) override;

    void reset() override;

    // Sync CPU-side int8 vectors to the persistent NPU buffer.
    // Call after bulk adds (e.g., from warm_ann_index).
    Expected<void> sync();

    // True if there are un-synced vectors pending upload.
    bool needs_sync() const;

    bool valid() const { return impl_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw

#endif // CATWHISPER_INDEX_FLAT_NPU_HPP
