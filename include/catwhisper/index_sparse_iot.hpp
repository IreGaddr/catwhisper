#ifndef CATWHISPER_INDEX_SPARSE_IOT_HPP
#define CATWHISPER_INDEX_SPARSE_IOT_HPP

#include <catwhisper/distance_iot.hpp>
#include <catwhisper/error.hpp>
#include <catwhisper/sparse_vector.hpp>
#include <catwhisper/types.hpp>

#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace cw {

// IndexSparseIOT: Fractal Binary Path Index for the Involuted Oblate Toroid.
//
// Instead of HNSW graph traversal on dense vectors, this index exploits the
// fractal self-similar structure of the IOT metric. Each stored vector
// generates a binary path signature against a set of reference points
// (pivots). The signature encodes which dimensions chose the involuted path
// for each pivot — a d-bit fractal fingerprint.
//
// Search: compute query's path signature against the same pivots, then
// find stored vectors with smallest Hamming distance on the signature.
// Hamming-adjacent signatures correspond to geometrically nearby points
// because the binary path choice is locally coherent on the IOT.
//
// Key properties:
//   - Dynamic dimensionality: vectors can have different effective dimensions.
//     No rebuild needed when dimensions grow.
//   - Sparse native: only non-zero entries are stored and compared.
//   - O(1) per-dimension path choice, O(nnz) per distance computation.
//   - Pivot signatures enable sub-linear search via multi-probe hashing.
//
// The pivot set grows adaptively: new pivots are added when existing pivots
// don't provide sufficient discrimination for newly added vectors.

struct SparseIOTParams {
    IOTParams iot;                     // IOT metric parameters
    uint32_t n_pivots = 32;           // Number of reference pivots
    uint32_t n_probes = 4;            // Multi-probe: check this many Hamming neighbors
    uint32_t signature_bits = 256;    // Bits per pivot signature (capped at n_active_dims)
    uint32_t pivot_sample_size = 128; // Sample size for pivot selection
};

class IndexSparseIOT {
public:
    [[nodiscard]] static Expected<IndexSparseIOT> create(
        const SparseIOTParams& params = {}
    );

    IndexSparseIOT() = default;
    IndexSparseIOT(IndexSparseIOT&&) noexcept;
    IndexSparseIOT& operator=(IndexSparseIOT&&) noexcept;
    ~IndexSparseIOT();

    // Add a single sparse vector with its ID.
    Expected<void> add(VectorId id, const SparseVector& vec);

    // Batch add.
    Expected<void> add_batch(std::span<const VectorId> ids,
                             std::span<const SparseVector> vecs);

    // Search: find k nearest neighbors to query.
    Expected<SearchResults> search(const SparseVector& query, uint32_t k);

    // Batch search.
    Expected<SearchResults> search_batch(std::span<const SparseVector> queries,
                                         uint32_t k);

    // Notify the index that dimensionality has grown.
    // This doesn't require rebuild — pivot signatures are recomputed lazily.
    void notify_dimension_growth(uint32_t new_max_dim);

    uint64_t size() const;
    uint32_t max_dimension() const;
    uint32_t n_pivots() const;

    Expected<void> save(const std::filesystem::path& path) const;
    Expected<void> load(const std::filesystem::path& path);

    void reset();
    bool valid() const { return impl_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw

#endif // CATWHISPER_INDEX_SPARSE_IOT_HPP
