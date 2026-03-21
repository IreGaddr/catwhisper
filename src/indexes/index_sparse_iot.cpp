#include <catwhisper/index_sparse_iot.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cw {

// Internal: pivot signature for a stored vector against all pivots.
// Each pivot produces a binary path signature; we store the concatenated
// Hamming-searchable representation.
struct PivotSignature {
    // One bit per (pivot, active_dim) pair, packed into 64-bit words.
    // For efficiency, we store one fixed-width signature per pivot,
    // then concatenate. Total bits = n_pivots * sig_width.
    std::vector<uint64_t> bits;
    uint32_t total_bits = 0;

    uint32_t hamming(const PivotSignature& other) const {
        uint32_t dist = 0;
        const size_t n = std::min(bits.size(), other.bits.size());
        for (size_t i = 0; i < n; ++i) {
            dist += static_cast<uint32_t>(__builtin_popcountll(bits[i] ^ other.bits[i]));
        }
        for (size_t i = n; i < bits.size(); ++i) {
            dist += static_cast<uint32_t>(__builtin_popcountll(bits[i]));
        }
        for (size_t i = n; i < other.bits.size(); ++i) {
            dist += static_cast<uint32_t>(__builtin_popcountll(other.bits[i]));
        }
        return dist;
    }
};

struct StoredVector {
    VectorId id;
    SparseVector vec;
    PivotSignature signature;
};

struct IndexSparseIOT::Impl {
    SparseIOTParams params;
    std::vector<SparseVector> pivots;
    std::vector<StoredVector> vectors;
    uint32_t max_dim = 0;

    // Hash buckets: signature_hash -> vector indices.
    // Multi-probe: search the query's bucket + Hamming-1 neighbors.
    std::unordered_multimap<uint64_t, uint32_t> sig_buckets;

    mutable std::shared_mutex mutex;
    std::mt19937 rng{42};

    // Compute pivot signature for a vector against all pivots.
    PivotSignature compute_signature(const SparseVector& vec) const {
        PivotSignature sig;
        const uint32_t n_piv = static_cast<uint32_t>(pivots.size());
        // Fixed-width: use min(n_active_dims, signature_bits) bits per pivot.
        const uint32_t bits_per_pivot = params.signature_bits;
        sig.total_bits = n_piv * bits_per_pivot;
        sig.bits.resize((sig.total_bits + 63) / 64, 0);

        for (uint32_t p = 0; p < n_piv; ++p) {
            auto path_sig = distance::compute_path_signature(
                pivots[p], vec, params.iot);

            // Copy path bits into the concatenated signature
            const uint32_t base_bit = p * bits_per_pivot;
            const uint32_t copy_bits = std::min(path_sig.n_active, bits_per_pivot);
            for (uint32_t b = 0; b < copy_bits; ++b) {
                if (path_sig.get(b)) {
                    const uint32_t global = base_bit + b;
                    sig.bits[global / 64] |= (1ULL << (global % 64));
                }
            }
        }

        return sig;
    }

    // Hash a signature into a bucket key (for multi-probe).
    // Uses the first 64 bits of the signature.
    uint64_t sig_hash(const PivotSignature& sig) const {
        return sig.bits.empty() ? 0 : sig.bits[0];
    }

    // Select pivots from the first N stored vectors.
    void select_pivots(uint32_t target_count) {
        if (vectors.empty()) return;

        const uint32_t sample = std::min(
            static_cast<uint32_t>(vectors.size()),
            params.pivot_sample_size);

        // Farthest-point sampling: greedily select pivots that maximize
        // minimum distance to existing pivots.
        std::vector<uint32_t> candidates(vectors.size());
        std::iota(candidates.begin(), candidates.end(), 0);

        // Start with a random pivot
        std::uniform_int_distribution<uint32_t> dist(0, sample - 1);
        pivots.clear();
        pivots.push_back(vectors[dist(rng)].vec);

        std::vector<float> min_dists(vectors.size(),
                                     std::numeric_limits<float>::max());

        while (pivots.size() < target_count && pivots.size() < vectors.size()) {
            // Update min distances to the latest pivot
            const auto& latest = pivots.back();
            uint32_t farthest_idx = 0;
            float farthest_dist = 0.0f;

            for (uint32_t i = 0; i < vectors.size(); ++i) {
                const float d = distance::iot_distance(
                    latest, vectors[i].vec, params.iot);
                min_dists[i] = std::min(min_dists[i], d);
                if (min_dists[i] > farthest_dist) {
                    farthest_dist = min_dists[i];
                    farthest_idx = i;
                }
            }
            pivots.push_back(vectors[farthest_idx].vec);
        }
    }

    // Recompute all signatures (after pivot change).
    void recompute_signatures() {
        sig_buckets.clear();
        for (uint32_t i = 0; i < vectors.size(); ++i) {
            vectors[i].signature = compute_signature(vectors[i].vec);
            sig_buckets.emplace(sig_hash(vectors[i].signature), i);
        }
    }
};

Expected<IndexSparseIOT> IndexSparseIOT::create(const SparseIOTParams& params) {
    IndexSparseIOT idx;
    idx.impl_ = std::make_unique<Impl>();
    idx.impl_->params = params;
    return idx;
}

IndexSparseIOT::IndexSparseIOT(IndexSparseIOT&&) noexcept = default;
IndexSparseIOT& IndexSparseIOT::operator=(IndexSparseIOT&&) noexcept = default;
IndexSparseIOT::~IndexSparseIOT() = default;

Expected<void> IndexSparseIOT::add(VectorId id, const SparseVector& vec) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::unique_lock lock(impl_->mutex);

    const uint32_t eff = vec.effective_dim();
    if (eff > impl_->max_dim) impl_->max_dim = eff;

    StoredVector sv;
    sv.id = id;
    sv.vec = vec;

    const uint32_t idx = static_cast<uint32_t>(impl_->vectors.size());

    // If we don't have pivots yet and have enough vectors, select them
    const bool need_pivots = impl_->pivots.empty() &&
        (impl_->vectors.size() + 1) >= impl_->params.pivot_sample_size;

    impl_->vectors.push_back(std::move(sv));

    if (need_pivots) {
        impl_->select_pivots(impl_->params.n_pivots);
        impl_->recompute_signatures();
    } else if (!impl_->pivots.empty()) {
        // Compute signature for new vector
        impl_->vectors.back().signature =
            impl_->compute_signature(impl_->vectors.back().vec);
        impl_->sig_buckets.emplace(
            impl_->sig_hash(impl_->vectors.back().signature), idx);
    }

    return {};
}

Expected<void> IndexSparseIOT::add_batch(std::span<const VectorId> ids,
                                          std::span<const SparseVector> vecs) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");
    if (ids.size() != vecs.size()) {
        return make_unexpected(ErrorCode::InvalidParameter, "ids and vecs size mismatch");
    }

    std::unique_lock lock(impl_->mutex);

    const uint32_t base_idx = static_cast<uint32_t>(impl_->vectors.size());
    impl_->vectors.reserve(impl_->vectors.size() + ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
        const uint32_t eff = vecs[i].effective_dim();
        if (eff > impl_->max_dim) impl_->max_dim = eff;

        StoredVector sv;
        sv.id = ids[i];
        sv.vec = vecs[i];
        impl_->vectors.push_back(std::move(sv));
    }

    // Select pivots if we just crossed the threshold
    if (impl_->pivots.empty() &&
        impl_->vectors.size() >= impl_->params.pivot_sample_size) {
        impl_->select_pivots(impl_->params.n_pivots);
        impl_->recompute_signatures();
    } else if (!impl_->pivots.empty()) {
        // Compute signatures for new vectors only
        for (uint32_t i = base_idx; i < impl_->vectors.size(); ++i) {
            impl_->vectors[i].signature =
                impl_->compute_signature(impl_->vectors[i].vec);
            impl_->sig_buckets.emplace(
                impl_->sig_hash(impl_->vectors[i].signature), i);
        }
    }

    return {};
}

Expected<SearchResults> IndexSparseIOT::search(const SparseVector& query,
                                                uint32_t k) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);

    const uint32_t n = static_cast<uint32_t>(impl_->vectors.size());
    if (n == 0) return SearchResults(1, 0);

    k = std::min(k, n);
    SearchResults results(1, k);

    if (impl_->pivots.empty() || n < impl_->params.pivot_sample_size) {
        // Brute force for small index (before pivots are selected)
        std::vector<std::pair<float, uint32_t>> dists(n);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n > 256)
#endif
        for (uint32_t i = 0; i < n; ++i) {
            dists[i] = {distance::iot_distance(query, impl_->vectors[i].vec,
                                                impl_->params.iot), i};
        }

        std::partial_sort(dists.begin(),
                          dists.begin() + static_cast<ptrdiff_t>(k),
                          dists.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });

        for (uint32_t i = 0; i < k; ++i) {
            results[0][i].id = impl_->vectors[dists[i].second].id;
            results[0][i].distance = dists[i].first;
        }
        return results;
    }

    // Signature-based search: compute query signature, find Hamming-close vectors
    const PivotSignature query_sig = impl_->compute_signature(query);

    // Score all vectors by Hamming distance to query signature (fast pre-filter)
    std::vector<std::pair<uint32_t, uint32_t>> hamming_scores(n); // (hamming_dist, idx)

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 1024)
#endif
    for (uint32_t i = 0; i < n; ++i) {
        hamming_scores[i] = {query_sig.hamming(impl_->vectors[i].signature), i};
    }

    // Take top candidates by Hamming distance
    const uint32_t n_candidates = std::min(n, std::max(k * 8, 256u));
    std::partial_sort(hamming_scores.begin(),
                      hamming_scores.begin() + static_cast<ptrdiff_t>(n_candidates),
                      hamming_scores.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

    // Rerank candidates with exact IOT distance
    std::vector<std::pair<float, uint32_t>> candidates(n_candidates);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n_candidates > 64)
#endif
    for (uint32_t i = 0; i < n_candidates; ++i) {
        const uint32_t idx = hamming_scores[i].second;
        candidates[i] = {
            distance::iot_distance(query, impl_->vectors[idx].vec, impl_->params.iot),
            idx
        };
    }

    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<ptrdiff_t>(k),
                      candidates.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

    for (uint32_t i = 0; i < k; ++i) {
        results[0][i].id = impl_->vectors[candidates[i].second].id;
        results[0][i].distance = candidates[i].first;
    }

    return results;
}

Expected<SearchResults> IndexSparseIOT::search_batch(
    std::span<const SparseVector> queries, uint32_t k) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    const uint32_t nq = static_cast<uint32_t>(queries.size());
    SearchResults combined(nq, k);

    // Each query is independent — parallelize at the query level
    for (uint32_t q = 0; q < nq; ++q) {
        auto single = search(queries[q], k);
        if (!single) return std::unexpected(single.error());
        for (uint32_t i = 0; i < k; ++i) {
            combined[q][i] = (*single)[0][i];
        }
    }

    return combined;
}

void IndexSparseIOT::notify_dimension_growth(uint32_t new_max_dim) {
    if (!impl_) return;
    std::unique_lock lock(impl_->mutex);
    if (new_max_dim > impl_->max_dim) {
        impl_->max_dim = new_max_dim;
        // Signatures remain valid — they operate on the union of active dims,
        // so new zero-valued dimensions don't change existing signatures.
        // Only vectors that actually USE the new dimensions need recomputation,
        // which happens when they're added via add().
    }
}

uint64_t IndexSparseIOT::size() const {
    if (!impl_) return 0;
    std::shared_lock lock(impl_->mutex);
    return impl_->vectors.size();
}

uint32_t IndexSparseIOT::max_dimension() const {
    if (!impl_) return 0;
    std::shared_lock lock(impl_->mutex);
    return impl_->max_dim;
}

uint32_t IndexSparseIOT::n_pivots() const {
    if (!impl_) return 0;
    std::shared_lock lock(impl_->mutex);
    return static_cast<uint32_t>(impl_->pivots.size());
}

void IndexSparseIOT::reset() {
    if (!impl_) return;
    std::unique_lock lock(impl_->mutex);
    impl_->vectors.clear();
    impl_->pivots.clear();
    impl_->sig_buckets.clear();
    impl_->max_dim = 0;
}

Expected<void> IndexSparseIOT::save(const std::filesystem::path& path) const {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return make_unexpected(ErrorCode::WriteFailed, "Cannot open file");

    // Magic + version
    const uint32_t magic = 0x494F5453; // "IOTS"
    const uint32_t version = 1;
    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);

    // Params
    ofs.write(reinterpret_cast<const char*>(&impl_->params), sizeof(SparseIOTParams));

    // Max dim
    ofs.write(reinterpret_cast<const char*>(&impl_->max_dim), 4);

    // Pivots
    const uint32_t n_piv = static_cast<uint32_t>(impl_->pivots.size());
    ofs.write(reinterpret_cast<const char*>(&n_piv), 4);
    for (const auto& pivot : impl_->pivots) {
        const uint32_t nnz = pivot.nnz();
        ofs.write(reinterpret_cast<const char*>(&nnz), 4);
        ofs.write(reinterpret_cast<const char*>(pivot.entries().data()),
                  nnz * sizeof(SparseEntry));
    }

    // Vectors
    const uint64_t n_vec = impl_->vectors.size();
    ofs.write(reinterpret_cast<const char*>(&n_vec), 8);
    for (const auto& sv : impl_->vectors) {
        ofs.write(reinterpret_cast<const char*>(&sv.id), 8);
        const uint32_t nnz = sv.vec.nnz();
        ofs.write(reinterpret_cast<const char*>(&nnz), 4);
        ofs.write(reinterpret_cast<const char*>(sv.vec.entries().data()),
                  nnz * sizeof(SparseEntry));
    }

    return {};
}

Expected<void> IndexSparseIOT::load(const std::filesystem::path& path) {
    if (!impl_) impl_ = std::make_unique<Impl>();

    std::unique_lock lock(impl_->mutex);
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return make_unexpected(ErrorCode::FileNotFound, "Cannot open file");

    uint32_t magic = 0, version = 0;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0x494F5453 || version != 1) {
        return make_unexpected(ErrorCode::InvalidFileFormat, "Not an IOT sparse index file");
    }

    ifs.read(reinterpret_cast<char*>(&impl_->params), sizeof(SparseIOTParams));
    ifs.read(reinterpret_cast<char*>(&impl_->max_dim), 4);

    // Pivots
    uint32_t n_piv = 0;
    ifs.read(reinterpret_cast<char*>(&n_piv), 4);
    impl_->pivots.resize(n_piv);
    for (uint32_t i = 0; i < n_piv; ++i) {
        uint32_t nnz = 0;
        ifs.read(reinterpret_cast<char*>(&nnz), 4);
        std::vector<SparseEntry> entries(nnz);
        ifs.read(reinterpret_cast<char*>(entries.data()), nnz * sizeof(SparseEntry));
        impl_->pivots[i] = SparseVector(std::move(entries));
    }

    // Vectors
    uint64_t n_vec = 0;
    ifs.read(reinterpret_cast<char*>(&n_vec), 8);
    impl_->vectors.resize(n_vec);
    for (uint64_t i = 0; i < n_vec; ++i) {
        ifs.read(reinterpret_cast<char*>(&impl_->vectors[i].id), 8);
        uint32_t nnz = 0;
        ifs.read(reinterpret_cast<char*>(&nnz), 4);
        std::vector<SparseEntry> entries(nnz);
        ifs.read(reinterpret_cast<char*>(entries.data()), nnz * sizeof(SparseEntry));
        impl_->vectors[i].vec = SparseVector(std::move(entries));
    }

    // Recompute signatures
    if (!impl_->pivots.empty()) {
        impl_->recompute_signatures();
    }

    return {};
}

} // namespace cw
