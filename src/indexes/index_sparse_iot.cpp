#include <catwhisper/index_sparse_iot.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <shared_mutex>
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
    // Total bits = n_pivots * sig_width.
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
};

// ============================================================================
// VP-tree for sub-linear Hamming nearest-neighbor search on PivotSignatures.
//
// Stored as a flat array in DFS order for cache locality. Left child of node i
// is always at i+1; right child index is stored explicitly. Leaf nodes have
// subtree_size == 1.
//
// Search prunes subtrees using the triangle inequality on Hamming distance:
//   - Left subtree contains vectors with hamming(vp, v) <= threshold
//   - Right subtree contains vectors with hamming(vp, v) > threshold
//   - We only need to search a subtree if it could contain a vector closer
//     than the current k-th best.
// ============================================================================
struct VPTree {
    struct Node {
        uint32_t sig_idx;       // Index into signatures array (vantage point)
        uint32_t threshold;     // Median Hamming distance for split
        uint32_t right_child;   // Index of right child in nodes[] (UINT32_MAX = none)
        uint32_t subtree_size;  // Number of vectors in subtree
    };

    std::vector<Node> nodes;

    bool empty() const { return nodes.empty(); }

    void clear() { nodes.clear(); }

    // Build from signature indices. Reorders `indices` in-place.
    void build(const std::vector<PivotSignature>& sigs,
               std::vector<uint32_t>& indices,
               std::mt19937& rng) {
        nodes.clear();
        if (indices.empty()) return;
        nodes.reserve(indices.size());
        build_recursive(sigs, indices, 0,
                        static_cast<uint32_t>(indices.size()), rng);
    }

    // Find k nearest signatures by Hamming distance.
    // Appends (hamming_dist, sig_idx) pairs to `results`.
    void search(const PivotSignature& query,
                const std::vector<PivotSignature>& sigs,
                uint32_t k,
                std::vector<std::pair<uint32_t, uint32_t>>& results) const {
        if (nodes.empty()) return;

        // Max-heap of (hamming_dist, sig_idx) — top is worst (largest distance)
        auto cmp = [](const std::pair<uint32_t, uint32_t>& a,
                      const std::pair<uint32_t, uint32_t>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<uint32_t, uint32_t>,
                            std::vector<std::pair<uint32_t, uint32_t>>,
                            decltype(cmp)> heap(cmp);

        uint32_t tau = UINT32_MAX; // Current search radius

        // DFS stack
        std::vector<uint32_t> stack;
        stack.reserve(64);
        stack.push_back(0);

        while (!stack.empty()) {
            const uint32_t ni = stack.back();
            stack.pop_back();

            if (ni == UINT32_MAX) continue;

            const auto& node = nodes[ni];
            const uint32_t d = query.hamming(sigs[node.sig_idx]);

            // Update result heap
            if (heap.size() < k || d < tau) {
                heap.push({d, node.sig_idx});
                if (heap.size() > k) heap.pop();
                if (heap.size() == k) tau = heap.top().first;
            }

            if (node.subtree_size == 1) continue; // Leaf

            // Left child at ni + 1: contains vectors with hamming(vp, v) <= threshold
            // Right child at node.right_child: contains vectors with hamming(vp, v) > threshold
            const uint32_t left_child = ni + 1;

            // Pruning via triangle inequality:
            // Left subtree has vectors with hamming(vp, v) <= threshold.
            //   Min possible hamming(q, v) = max(0, d - threshold).
            //   Search if max(0, d - threshold) <= tau.
            //   Overflow-safe: (d <= threshold) || (d - threshold <= tau).
            // Right subtree has vectors with hamming(vp, v) > threshold.
            //   Min possible hamming(q, v) = max(0, threshold + 1 - d).
            //   Search if (d > threshold) || (threshold - d < tau).

            const bool search_left = (d <= node.threshold) ||
                                      (d - node.threshold <= tau);
            const bool search_right = (node.right_child != UINT32_MAX) &&
                                       (d > node.threshold || node.threshold - d < tau);

            // Push less promising subtree first (popped last = searched last)
            if (d <= node.threshold) {
                // Query closer to left subtree — search left first (push right, then left)
                if (search_right) stack.push_back(node.right_child);
                if (search_left)  stack.push_back(left_child);
            } else {
                // Query closer to right subtree — search right first
                if (search_left)  stack.push_back(left_child);
                if (search_right) stack.push_back(node.right_child);
            }
        }

        // Extract results (closest first)
        const size_t n_results = heap.size();
        results.resize(n_results);
        for (size_t i = n_results; i > 0; --i) {
            results[i - 1] = heap.top();
            heap.pop();
        }
    }

private:
    void build_recursive(const std::vector<PivotSignature>& sigs,
                         std::vector<uint32_t>& indices,
                         uint32_t begin, uint32_t end,
                         std::mt19937& rng) {
        if (begin >= end) return;

        const uint32_t count = end - begin;
        const uint32_t node_pos = static_cast<uint32_t>(nodes.size());
        nodes.push_back({});

        if (count == 1) {
            nodes[node_pos] = {indices[begin], 0, UINT32_MAX, 1};
            return;
        }

        // Select random vantage point
        std::uniform_int_distribution<uint32_t> dist(begin, end - 1);
        const uint32_t vp_pos = dist(rng);
        std::swap(indices[begin], indices[vp_pos]);
        const uint32_t vp = indices[begin];

        // Compute Hamming distances from VP to all other vectors in range
        // and partition around the median distance.
        const uint32_t n_rest = count - 1;
        const uint32_t median_pos = begin + 1 + n_rest / 2;

        // nth_element to find median by Hamming distance to VP
        std::nth_element(
            indices.begin() + begin + 1,
            indices.begin() + median_pos,
            indices.begin() + end,
            [&](uint32_t a, uint32_t b) {
                return sigs[a].hamming(sigs[vp]) < sigs[b].hamming(sigs[vp]);
            }
        );

        const uint32_t threshold = sigs[vp].hamming(sigs[indices[median_pos]]);

        nodes[node_pos] = {vp, threshold, UINT32_MAX, count};

        // Build left subtree (vectors closer to VP): indices[begin+1..median_pos)
        build_recursive(sigs, indices, begin + 1, median_pos, rng);

        // Build right subtree (vectors farther from VP): indices[median_pos..end)
        nodes[node_pos].right_child = static_cast<uint32_t>(nodes.size());
        build_recursive(sigs, indices, median_pos, end, rng);
    }
};

// ============================================================================

struct IndexSparseIOT::Impl {
    SparseIOTParams params;
    std::vector<SparseVector> pivots;
    std::vector<StoredVector> vectors;

    // Signatures stored separately from vectors for cache-friendly Hamming scan
    // and VP-tree access. signatures[i] corresponds to vectors[i].
    std::vector<PivotSignature> signatures;

    // VP-tree over signatures for sub-linear Hamming nearest-neighbor.
    // Built from all signatures present at build time.
    VPTree vp_tree;

    // Vectors added after the last VP-tree build. Searched via linear scan.
    // When pending grows large enough, the VP-tree is rebuilt.
    std::vector<uint32_t> pending_indices; // Indices into signatures[]

    static constexpr uint32_t kVPTreeRebuildThreshold = 512;

    uint32_t max_dim = 0;

    mutable std::shared_mutex mutex;
    std::mt19937 rng{42};

    // Compute pivot signature for a vector against all pivots.
    PivotSignature compute_signature(const SparseVector& vec) const {
        PivotSignature sig;
        const uint32_t n_piv = static_cast<uint32_t>(pivots.size());
        const uint32_t bits_per_pivot = params.signature_bits;
        sig.total_bits = n_piv * bits_per_pivot;
        sig.bits.resize((sig.total_bits + 63) / 64, 0);

        for (uint32_t p = 0; p < n_piv; ++p) {
            auto path_sig = distance::compute_path_signature(
                pivots[p], vec, params.iot);

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

    // Select pivots via farthest-point sampling.
    void select_pivots(uint32_t target_count) {
        if (vectors.empty()) return;

        std::vector<uint32_t> candidates(vectors.size());
        std::iota(candidates.begin(), candidates.end(), 0);

        std::uniform_int_distribution<uint32_t> dist(
            0, std::min(static_cast<uint32_t>(vectors.size()),
                        params.pivot_sample_size) - 1);
        pivots.clear();
        pivots.push_back(vectors[dist(rng)].vec);

        std::vector<float> min_dists(vectors.size(),
                                     std::numeric_limits<float>::max());

        while (pivots.size() < target_count && pivots.size() < vectors.size()) {
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

    // Recompute all signatures and rebuild VP-tree.
    void recompute_signatures() {
        const uint32_t n = static_cast<uint32_t>(vectors.size());
        signatures.resize(n);

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n > 64)
#endif
        for (uint32_t i = 0; i < n; ++i) {
            signatures[i] = compute_signature(vectors[i].vec);
        }

        rebuild_vp_tree();
    }

    // Rebuild VP-tree from all current signatures.
    void rebuild_vp_tree() {
        pending_indices.clear();
        const uint32_t n = static_cast<uint32_t>(signatures.size());
        if (n == 0) {
            vp_tree.clear();
            return;
        }

        std::vector<uint32_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        vp_tree.build(signatures, indices, rng);
    }

    // Search the VP-tree + pending buffer for Hamming-nearest signatures.
    // Returns (hamming_dist, vector_index) pairs.
    void hamming_search(const PivotSignature& query_sig, uint32_t k_candidates,
                        std::vector<std::pair<uint32_t, uint32_t>>& candidates) const {
        candidates.clear();

        // Phase 1: VP-tree search (sub-linear)
        if (!vp_tree.empty()) {
            vp_tree.search(query_sig, signatures, k_candidates, candidates);
        }

        // Phase 2: Linear scan of pending buffer
        for (const uint32_t idx : pending_indices) {
            const uint32_t h = query_sig.hamming(signatures[idx]);
            candidates.push_back({h, idx});
        }

        // If VP-tree returned results + pending, re-sort and truncate
        if (candidates.size() > k_candidates) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + static_cast<ptrdiff_t>(k_candidates),
                candidates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            candidates.resize(k_candidates);
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

    const bool need_pivots = impl_->pivots.empty() &&
        (impl_->vectors.size() + 1) >= impl_->params.pivot_sample_size;

    impl_->vectors.push_back(std::move(sv));

    if (need_pivots) {
        impl_->select_pivots(impl_->params.n_pivots);
        impl_->recompute_signatures();
    } else if (!impl_->pivots.empty()) {
        // Compute signature for new vector and add to pending
        impl_->signatures.push_back(
            impl_->compute_signature(impl_->vectors.back().vec));
        impl_->pending_indices.push_back(idx);

        // Rebuild VP-tree if pending buffer is large
        if (impl_->pending_indices.size() >= Impl::kVPTreeRebuildThreshold) {
            impl_->rebuild_vp_tree();
        }
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
        // Compute signatures for new vectors in parallel
        const uint32_t n_new = static_cast<uint32_t>(impl_->vectors.size()) - base_idx;
        const size_t old_sig_size = impl_->signatures.size();
        impl_->signatures.resize(old_sig_size + n_new);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 256) if(n_new > 256)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(n_new); ++i) {
            impl_->signatures[old_sig_size + static_cast<size_t>(i)] =
                impl_->compute_signature(impl_->vectors[base_idx + static_cast<uint32_t>(i)].vec);
        }

        for (uint32_t i = base_idx; i < impl_->vectors.size(); ++i) {
            impl_->pending_indices.push_back(i);
        }

        // Batch add: always rebuild VP-tree
        impl_->rebuild_vp_tree();
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

    // Signature-based search via VP-tree + pending scan
    const PivotSignature query_sig = impl_->compute_signature(query);

    const uint32_t n_candidates = std::min(n, std::max(k * 8, 256u));

    std::vector<std::pair<uint32_t, uint32_t>> hamming_candidates;
    impl_->hamming_search(query_sig, n_candidates, hamming_candidates);

    // Rerank candidates with exact IOT distance
    const uint32_t n_rerank = static_cast<uint32_t>(hamming_candidates.size());
    std::vector<std::pair<float, uint32_t>> candidates(n_rerank);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n_rerank > 64)
#endif
    for (uint32_t i = 0; i < n_rerank; ++i) {
        const uint32_t idx = hamming_candidates[i].second;
        candidates[i] = {
            distance::iot_distance(query, impl_->vectors[idx].vec, impl_->params.iot),
            idx
        };
    }

    const uint32_t k_final = std::min(k, n_rerank);
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<ptrdiff_t>(k_final),
                      candidates.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

    for (uint32_t i = 0; i < k_final; ++i) {
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
    impl_->signatures.clear();
    impl_->vp_tree.clear();
    impl_->pending_indices.clear();
    impl_->max_dim = 0;
}

Expected<void> IndexSparseIOT::save(const std::filesystem::path& path) const {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return make_unexpected(ErrorCode::WriteFailed, "Cannot open file");

    // Magic + version (v2: VP-tree format, no sig_buckets)
    const uint32_t magic = 0x494F5453; // "IOTS"
    const uint32_t version = 2;
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

    // Vectors (sparse data only — signatures recomputed on load)
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
    if (magic != 0x494F5453 || (version != 1 && version != 2)) {
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

    // Recompute signatures + build VP-tree
    if (!impl_->pivots.empty()) {
        impl_->recompute_signatures();
    }

    return {};
}

} // namespace cw
