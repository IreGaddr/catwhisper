#include <catwhisper/index_ivf_flat.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>
#include "core/context_impl.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <random>
#include <limits>
#include <set>

namespace cw {

// K-means clustering implementation (CPU)
class KMeans {
public:
    KMeans(uint32_t n_clusters, uint32_t dimension, uint32_t max_iters, Metric metric)
        : n_clusters_(n_clusters), dimension_(dimension), max_iters_(max_iters), metric_(metric) {}

    std::vector<float> fit(const float* data, uint64_t n_samples) {
        if (n_samples < n_clusters_) {
            n_clusters_ = static_cast<uint32_t>(n_samples);
        }

        std::vector<float> centroids(n_clusters_ * dimension_);
        kmeans_plusplus_init(data, n_samples, centroids.data());

        std::vector<uint32_t> assignments(n_samples);

        for (uint32_t iter = 0; iter < max_iters_; ++iter) {
            assign_clusters(data, n_samples, centroids.data(), assignments.data());
            bool converged = update_centroids(data, n_samples, assignments.data(), centroids.data());
            if (converged) break;
        }

        return centroids;
    }

    uint32_t nclusters() const { return n_clusters_; }

private:
    uint32_t n_clusters_;
    uint32_t dimension_;
    uint32_t max_iters_;
    Metric metric_;

    void kmeans_plusplus_init(const float* data, uint64_t n_samples, float* centroids) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint64_t> uniform_dist(0, n_samples - 1);

        uint64_t first_idx = uniform_dist(rng);
        std::memcpy(centroids, data + first_idx * dimension_, dimension_ * sizeof(float));

        std::vector<double> min_distances(n_samples, std::numeric_limits<double>::max());

        for (uint32_t c = 1; c < n_clusters_; ++c) {
            double total_dist = 0.0;
            for (uint64_t i = 0; i < n_samples; ++i) {
                double dist = compute_distance(data + i * dimension_, centroids + (c - 1) * dimension_);
                if (dist < min_distances[i]) {
                    min_distances[i] = dist;
                }
                total_dist += min_distances[i];
            }

            std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
            double threshold = prob_dist(rng);
            double cumulative = 0.0;
            uint64_t next_idx = 0;
            for (uint64_t i = 0; i < n_samples; ++i) {
                cumulative += min_distances[i];
                if (cumulative >= threshold) {
                    next_idx = i;
                    break;
                }
            }

            std::memcpy(centroids + c * dimension_, data + next_idx * dimension_, dimension_ * sizeof(float));
        }
    }

    double compute_distance(const float* a, const float* b) const {
        double dist = 0.0;
        if (metric_ == Metric::L2) {
            for (uint32_t d = 0; d < dimension_; ++d) {
                double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
                dist += diff * diff;
            }
        } else {
            for (uint32_t d = 0; d < dimension_; ++d) {
                dist += static_cast<double>(a[d]) * static_cast<double>(b[d]);
            }
            dist = -dist;
        }
        return dist;
    }

    void assign_clusters(const float* data, uint64_t n_samples,
                         const float* centroids, uint32_t* assignments) {
        for (uint64_t i = 0; i < n_samples; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            uint32_t best_cluster = 0;

            for (uint32_t c = 0; c < n_clusters_; ++c) {
                double dist = compute_distance(data + i * dimension_, centroids + c * dimension_);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
    }

    bool update_centroids(const float* data, uint64_t n_samples,
                          const uint32_t* assignments, float* centroids) {
        std::vector<double> new_centroids(n_clusters_ * dimension_, 0.0);
        std::vector<uint64_t> counts(n_clusters_, 0);

        for (uint64_t i = 0; i < n_samples; ++i) {
            uint32_t c = assignments[i];
            counts[c]++;
            for (uint32_t d = 0; d < dimension_; ++d) {
                new_centroids[c * dimension_ + d] += static_cast<double>(data[i * dimension_ + d]);
            }
        }

        bool converged = true;
        for (uint32_t c = 0; c < n_clusters_; ++c) {
            if (counts[c] > 0) {
                for (uint32_t d = 0; d < dimension_; ++d) {
                    float new_val = static_cast<float>(new_centroids[c * dimension_ + d] / counts[c]);
                    if (std::abs(new_val - centroids[c * dimension_ + d]) > 1e-6f) {
                        converged = false;
                    }
                    centroids[c * dimension_ + d] = new_val;
                }
            }
        }

        return converged;
    }
};

struct IndexIVFFlat::Impl {
    Context* ctx = nullptr;
    uint32_t dimension = 0;
    uint64_t n_vectors = 0;
    Metric metric = Metric::L2;
    bool use_fp16 = true;

    IVFParams params;

    // Trained centroids (nlist * dimension floats)
    std::vector<float> centroids;
    uint32_t actual_nlist = 0;

    // For CPU-based search, store vectors cluster-by-cluster
    // invlists_data[c] = raw vector data for cluster c (dimension floats each)
    std::vector<std::vector<float>> invlists_data;
    // invlists_ids[c] = vector IDs for cluster c
    std::vector<std::vector<VectorId>> invlists_ids;

    // ID mapping for external IDs
    std::vector<VectorId> id_mapping;

    bool is_trained = false;
};

IndexIVFFlat::IndexIVFFlat(IndexIVFFlat&& other) noexcept
    : impl_(std::move(other.impl_)) {}

IndexIVFFlat& IndexIVFFlat::operator=(IndexIVFFlat&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

IndexIVFFlat::~IndexIVFFlat() = default;

Expected<IndexIVFFlat> IndexIVFFlat::create(
    Context& ctx, uint32_t dimension,
    const IVFParams& params, const IndexOptions& options) {

    IndexIVFFlat index;
    index.impl_ = std::make_unique<Impl>();
    index.impl_->ctx = &ctx;
    index.impl_->dimension = dimension;
    index.impl_->metric = options.metric;
    index.impl_->use_fp16 = options.use_fp16;
    index.impl_->params = params;

    // Pre-allocate inverted lists
    index.impl_->invlists_data.resize(params.nlist);
    index.impl_->invlists_ids.resize(params.nlist);

    return index;
}

uint32_t IndexIVFFlat::dimension() const {
    return impl_ ? impl_->dimension : 0;
}

uint64_t IndexIVFFlat::size() const {
    return impl_ ? impl_->n_vectors : 0;
}

bool IndexIVFFlat::is_trained() const {
    return impl_ ? impl_->is_trained : false;
}

uint32_t IndexIVFFlat::nlist() const {
    return impl_ ? impl_->actual_nlist : 0;
}

uint32_t IndexIVFFlat::nprobe() const {
    return impl_ ? impl_->params.nprobe : 0;
}

void IndexIVFFlat::set_nprobe(uint32_t nprobe) {
    if (impl_) {
        impl_->params.nprobe = std::min(nprobe, impl_->actual_nlist);
    }
}

IndexStats IndexIVFFlat::stats() const {
    IndexStats s{};
    if (impl_) {
        s.n_vectors = impl_->n_vectors;
        s.dimension = impl_->dimension;
        s.is_trained = impl_->is_trained;
    }
    return s;
}

Expected<void> IndexIVFFlat::train(std::span<const float> data, uint64_t n) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) {
        return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");
    }

    // Run K-means clustering
    KMeans kmeans(impl_->params.nlist, impl_->dimension, impl_->params.kmeans_iters, impl_->metric);
    impl_->centroids = kmeans.fit(data.data(), n);
    impl_->actual_nlist = kmeans.nclusters();

    // Ensure nprobe doesn't exceed actual_nlist
    impl_->params.nprobe = std::min(impl_->params.nprobe, impl_->actual_nlist);

    impl_->is_trained = true;
    return {};
}

Expected<void> IndexIVFFlat::add(std::span<const float> data, uint64_t n,
                                  std::span<const VectorId> ids) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    if (!impl_->is_trained) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index must be trained before adding vectors");
    }

    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) {
        return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");
    }

    // Assign each new vector to nearest centroid
    for (uint64_t i = 0; i < n; ++i) {
        const float* vec = data.data() + i * impl_->dimension;

        double min_dist = std::numeric_limits<double>::max();
        uint32_t best_cluster = 0;

        for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
            double dist = 0.0;
            if (impl_->metric == Metric::L2) {
                for (uint32_t d = 0; d < impl_->dimension; ++d) {
                    double diff = static_cast<double>(vec[d]) - static_cast<double>(impl_->centroids[c * impl_->dimension + d]);
                    dist += diff * diff;
                }
            } else {
                for (uint32_t d = 0; d < impl_->dimension; ++d) {
                    dist += static_cast<double>(vec[d]) * static_cast<double>(impl_->centroids[c * impl_->dimension + d]);
                }
                dist = -dist;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        // Store vector data in cluster
        impl_->invlists_data[best_cluster].insert(
            impl_->invlists_data[best_cluster].end(),
            vec,
            vec + impl_->dimension
        );

        // Store ID
        VectorId external_id = ids.empty() ? (impl_->n_vectors + i) : ids[i];
        impl_->invlists_ids[best_cluster].push_back(external_id);
        impl_->id_mapping.push_back(external_id);
    }

    impl_->n_vectors += n;
    return {};
}

Expected<SearchResults> IndexIVFFlat::search(Vector query, uint32_t k) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    if (query.size() != impl_->dimension) {
        return make_unexpected(ErrorCode::InvalidDimension, "Query dimension mismatch");
    }

    if (impl_->n_vectors == 0) {
        return SearchResults(1, k);
    }

    // Find nprobe nearest centroids
    std::vector<std::pair<float, uint32_t>> centroid_distances;
    centroid_distances.reserve(impl_->actual_nlist);

    for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
        float dist = 0.0f;
        if (impl_->metric == Metric::L2) {
            for (uint32_t d = 0; d < impl_->dimension; ++d) {
                float diff = query[d] - impl_->centroids[c * impl_->dimension + d];
                dist += diff * diff;
            }
        } else {
            for (uint32_t d = 0; d < impl_->dimension; ++d) {
                dist += query[d] * impl_->centroids[c * impl_->dimension + d];
            }
            dist = -dist;
        }
        centroid_distances.emplace_back(dist, c);
    }

    // Sort by distance and take nprobe closest
    uint32_t nprobe = std::min(impl_->params.nprobe, impl_->actual_nlist);
    std::partial_sort(centroid_distances.begin(), centroid_distances.begin() + nprobe,
                      centroid_distances.end());

    // Collect candidates from selected clusters
    std::vector<std::pair<float, VectorId>> candidates;

    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster = centroid_distances[p].second;

        const auto& cluster_data = impl_->invlists_data[cluster];
        const auto& cluster_ids = impl_->invlists_ids[cluster];
        uint32_t cluster_size = static_cast<uint32_t>(cluster_ids.size());

        for (uint32_t v = 0; v < cluster_size; ++v) {
            const float* vec = cluster_data.data() + v * impl_->dimension;

            float dist = 0.0f;
            if (impl_->metric == Metric::L2) {
                for (uint32_t d = 0; d < impl_->dimension; ++d) {
                    float diff = query[d] - vec[d];
                    dist += diff * diff;
                }
            } else {
                for (uint32_t d = 0; d < impl_->dimension; ++d) {
                    dist += query[d] * vec[d];
                }
                dist = -dist;
            }

            candidates.emplace_back(dist, cluster_ids[v]);
        }
    }

    // Sort candidates and take top-k
    SearchResults results(1, k);
    uint32_t actual_k = std::min(k, static_cast<uint32_t>(candidates.size()));

    if (actual_k > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + actual_k, candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        for (uint32_t i = 0; i < actual_k; ++i) {
            results.results[i].distance = candidates[i].first;
            results.results[i].id = candidates[i].second;
        }
    }

    return results;
}

Expected<SearchResults> IndexIVFFlat::search(std::span<const float> queries,
                                              uint64_t n_queries, uint32_t k) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    if (impl_->n_vectors == 0) {
        return SearchResults(n_queries, k);
    }

    SearchResults results(n_queries, k);

    for (uint64_t q = 0; q < n_queries; ++q) {
        std::vector<float> query(queries.begin() + q * impl_->dimension,
                                  queries.begin() + (q + 1) * impl_->dimension);
        auto single_result = search(query, k);
        if (!single_result) {
            return single_result;
        }

        for (uint32_t i = 0; i < k; ++i) {
            results.results[q * k + i] = single_result->results[i];
        }
    }

    return results;
}

Expected<void> IndexIVFFlat::save(const std::filesystem::path& path) const {
    (void)path;
    return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

Expected<void> IndexIVFFlat::load(const std::filesystem::path& path) {
    (void)path;
    return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

void IndexIVFFlat::reset() {
    if (impl_) {
        impl_->n_vectors = 0;
        impl_->id_mapping.clear();
        for (auto& list : impl_->invlists_data) {
            list.clear();
        }
        for (auto& list : impl_->invlists_ids) {
            list.clear();
        }
    }
}

} // namespace cw
