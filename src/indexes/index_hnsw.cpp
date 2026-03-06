#include <catwhisper/index_hnsw.hpp>
#include <catwhisper/distance.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_set>
#include <random>
#include <shared_mutex>
#include <thread>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cw {

static constexpr uint32_t MAX_LAYERS = 64;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;
static constexpr size_t BUILD_LOCK_STRIPES = 8192;
using BuildLockArray = std::array<std::shared_mutex, BUILD_LOCK_STRIPES>;

struct Node {
    uint32_t id;
    uint32_t level;
    std::vector<std::vector<uint32_t>> neighbors;
};

struct IndexHNSW::Impl {
    uint32_t dimension = 0;
    uint64_t n_vectors = 0;
    Metric metric = Metric::L2;

    HNSWParams params;
    std::vector<Node> nodes;
    std::vector<float> data;
    std::vector<VectorId> id_mapping;
    std::unordered_set<VectorId> id_set;

    uint32_t entry_point = INVALID_NODE;
    uint32_t max_level = 0;

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> level_dist{0.0f, 1.0f};
    mutable std::shared_mutex mutex;
    
    Context* ctx = nullptr;
    HNSWGPUOptions gpu_options;
    bool gpu_initialized = false;
    
    Buffer gpu_data_buffer;
    Buffer gpu_query_buffer;
    Buffer gpu_distance_buffer;
    Pipeline gpu_distance_pipeline;
    DescriptorSet gpu_distance_desc_set;

    float distance(const float* a, const float* b) const {
#if defined(__AVX512F__)
        float result = 0.0f;
        const float* end = a + dimension;
        
        __m512 sum = _mm512_setzero_ps();
        
        while (a + 16 <= end) {
            __m512 va = _mm512_loadu_ps(a);
            __m512 vb = _mm512_loadu_ps(b);
            __m512 diff = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(diff, diff, sum);
            a += 16;
            b += 16;
        }
        
        result = _mm512_reduce_add_ps(sum);
        
        while (a < end) {
            float d = *a - *b;
            result += d * d;
            ++a;
            ++b;
        }
        
        if (metric == Metric::IP) {
            return -result;
        }
        return result;
#elif defined(__AVX2__)
        float result = 0.0f;
        const float* end = a + dimension;
        
        __m256 sum = _mm256_setzero_ps();
        
        while (a + 8 <= end) {
            __m256 va = _mm256_loadu_ps(a);
            __m256 vb = _mm256_loadu_ps(b);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            a += 8;
            b += 8;
        }
        
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum);
        for (int i = 0; i < 8; ++i) result += temp[i];
        
        while (a < end) {
            float d = *a - *b;
            result += d * d;
            ++a;
            ++b;
        }
        
        if (metric == Metric::IP) {
            return -result;
        }
        return result;
#else
        if (metric == Metric::IP) {
            return -distance::inner_product({a, dimension}, {b, dimension});
        }
        return distance::l2_sqr({a, dimension}, {b, dimension});
#endif
    }

    const float* get_vector(uint32_t node_id) const {
        return data.data() + static_cast<size_t>(node_id) * dimension;
    }

    uint32_t random_level() {
        float r = level_dist(rng);
        int level = static_cast<int>(-std::log(r) * params.ml_factor);
        return static_cast<uint32_t>(std::min(level, static_cast<int>(MAX_LAYERS - 1)));
    }

    // Min-heap: smallest distance at top
    using MinHeap = std::priority_queue<std::pair<float, uint32_t>,
                                        std::vector<std::pair<float, uint32_t>>,
                                        std::greater<>>;
    // Max-heap: largest distance at top
    using MaxHeap = std::priority_queue<std::pair<float, uint32_t>>;

    std::vector<std::pair<float, uint32_t>>
    search_layer(const float* query, uint32_t layer, uint32_t ef,
                 const std::vector<uint32_t>& entry_points,
                 BuildLockArray* build_locks = nullptr) {
        std::unordered_set<uint32_t> visited;
        visited.reserve(ef * 4);

        MinHeap candidates;  // min-heap: smallest dist on top
        MaxHeap results;     // max-heap: largest dist on top
        std::vector<uint32_t> neighbors_snapshot;

        for (uint32_t ep : entry_points) {
            float dist = distance(query, get_vector(ep));
            visited.insert(ep);
            candidates.emplace(dist, ep);
            results.emplace(dist, ep);
        }

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();

            // Don't terminate early - explore all reachable candidates for better recall
            // Original: if (c_dist > f_dist && results.size() >= ef) break;

            if (build_locks == nullptr) {
                const Node& node = nodes[c_id];
                if (layer >= node.neighbors.size()) continue;

                for (uint32_t neighbor : node.neighbors[layer]) {
                    if (visited.count(neighbor)) continue;
                    visited.insert(neighbor);

                    float n_dist = distance(query, get_vector(neighbor));
                    float f_dist_now = results.empty() ? INFINITY : results.top().first;

                    if (n_dist < f_dist_now || results.size() < ef) {
                        candidates.emplace(n_dist, neighbor);
                        results.emplace(n_dist, neighbor);

                        while (results.size() > ef) {
                            results.pop();
                        }
                    }
                }
                continue;
            }

            neighbors_snapshot.clear();
            {
                auto& stripe_lock =
                    (*build_locks)[static_cast<size_t>(c_id) & (BUILD_LOCK_STRIPES - 1)];
                std::shared_lock<std::shared_mutex> guard(stripe_lock);
                const Node& node = nodes[c_id];
                if (layer < node.neighbors.size()) {
                    const auto& nbrs = node.neighbors[layer];
                    neighbors_snapshot.assign(nbrs.begin(), nbrs.end());
                }
            }

            for (uint32_t neighbor : neighbors_snapshot) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);

                float n_dist = distance(query, get_vector(neighbor));
                float f_dist_now = results.empty() ? INFINITY : results.top().first;

                if (n_dist < f_dist_now || results.size() < ef) {
                    candidates.emplace(n_dist, neighbor);
                    results.emplace(n_dist, neighbor);

                    while (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<std::pair<float, uint32_t>> result_vec;
        result_vec.reserve(results.size());
        while (!results.empty()) {
            result_vec.push_back(results.top());
            results.pop();
        }
        std::reverse(result_vec.begin(), result_vec.end());

        return result_vec;
    }

    void select_neighbors_simple(std::vector<std::pair<float, uint32_t>>& candidates, uint32_t M) {
        if (candidates.size() <= M) return;
        std::partial_sort(candidates.begin(), candidates.begin() + M, candidates.end());
        candidates.resize(M);
    }

    void select_neighbors_heuristic(uint32_t node_id, std::vector<std::pair<float, uint32_t>>& candidates,
                                    uint32_t M) {
        if (candidates.size() <= M) return;

        std::sort(candidates.begin(), candidates.end());

        std::vector<std::pair<float, uint32_t>> selected;
        selected.reserve(M);

        for (const auto& [c_dist, c_id] : candidates) {
            if (selected.size() >= M) break;

            bool good = true;
            const float* c_vec = get_vector(c_id);

            for (const auto& [s_dist, s_id] : selected) {
                float d = distance(c_vec, get_vector(s_id));
                if (d < c_dist) {
                    good = false;
                    break;
                }
            }

            if (good) {
                selected.emplace_back(c_dist, c_id);
            }
        }

        // If heuristic selected fewer than M, fill with closest remaining
        for (const auto& [c_dist, c_id] : candidates) {
            if (selected.size() >= M) break;
            bool already_selected = false;
            for (const auto& [_, sid] : selected) {
                if (sid == c_id) { already_selected = true; break; }
            }
            if (!already_selected) {
                selected.emplace_back(c_dist, c_id);
            }
        }

        candidates = std::move(selected);
    }

    void shrink_connections(uint32_t node_id, uint32_t layer, uint32_t max_conn) {
        Node& node = nodes[node_id];
        if (layer >= node.neighbors.size() || node.neighbors[layer].size() <= max_conn) return;

        std::vector<std::pair<float, uint32_t>> candidates;
        const float* node_vec = get_vector(node_id);
        for (uint32_t n : node.neighbors[layer]) {
            candidates.emplace_back(distance(node_vec, get_vector(n)), n);
        }

        select_neighbors_simple(candidates, max_conn);

        node.neighbors[layer].clear();
        for (const auto& [_, n] : candidates) {
            node.neighbors[layer].push_back(n);
        }
    }

    void shrink_connections_heuristic(uint32_t node_id, uint32_t layer, uint32_t max_conn) {
        Node& node = nodes[node_id];
        if (layer >= node.neighbors.size() || node.neighbors[layer].size() <= max_conn) return;

        std::vector<std::pair<float, uint32_t>> candidates;
        const float* node_vec = get_vector(node_id);
        for (uint32_t n : node.neighbors[layer]) {
            candidates.emplace_back(distance(node_vec, get_vector(n)), n);
        }

        select_neighbors_heuristic(node_id, candidates, max_conn);

        node.neighbors[layer].clear();
        for (const auto& [_, n] : candidates) {
            node.neighbors[layer].push_back(n);
        }
    }

    void add_connection(uint32_t node_a, uint32_t node_b, uint32_t layer) {
        Node& a = nodes[node_a];
        if (layer >= a.neighbors.size()) {
            a.neighbors.resize(layer + 1);
        }
        a.neighbors[layer].push_back(node_b);
    }

    Expected<void> insert_node(const float* vec, VectorId external_id) {
        std::unique_lock lock(mutex);

        uint32_t node_id = static_cast<uint32_t>(n_vectors);

        if (id_set.count(external_id)) {
            return make_unexpected(ErrorCode::InvalidParameter, "Duplicate ID");
        }

        data.insert(data.end(), vec, vec + dimension);
        id_mapping.push_back(external_id);
        id_set.insert(external_id);

        uint32_t level = random_level();
        level = std::min(level, MAX_LAYERS - 1);

        Node node;
        node.id = node_id;
        node.level = level;
        node.neighbors.resize(level + 1);
        nodes.push_back(std::move(node));

        if (entry_point == INVALID_NODE) {
            entry_point = node_id;
            max_level = level;
            n_vectors++;
            return {};
        }

        std::vector<uint32_t> ep_set = {entry_point};

        for (int lc = static_cast<int>(max_level); lc > static_cast<int>(level); --lc) {
            auto results = search_layer(vec, static_cast<uint32_t>(lc), 1, ep_set);
            if (!results.empty()) {
                ep_set = {results[0].second};
            }
        }

        for (int lc = std::min(static_cast<int>(level), static_cast<int>(max_level)); lc >= 0; --lc) {
            auto results = search_layer(vec, static_cast<uint32_t>(lc), params.ef_construction, ep_set);

            uint32_t max_conn = (lc == 0) ? params.M * 2 : params.M;

            std::vector<std::pair<float, uint32_t>> neighbors;
            for (const auto& r : results) {
                neighbors.push_back(r);
            }

            select_neighbors_simple(neighbors, max_conn);

            nodes[node_id].neighbors[lc].reserve(neighbors.size());
            for (const auto& [_, n_id] : neighbors) {
                nodes[node_id].neighbors[lc].push_back(n_id);
                add_connection(n_id, node_id, static_cast<uint32_t>(lc));
                shrink_connections(n_id, static_cast<uint32_t>(lc), max_conn);
            }

            if (!results.empty()) {
                ep_set.clear();
                for (const auto& r : results) {
                    ep_set.push_back(r.second);
                }
            }
        }

        if (level > max_level) {
            max_level = level;
            entry_point = node_id;
        }

        n_vectors++;
        return {};
    }
};

IndexHNSW::IndexHNSW(IndexHNSW&& other) noexcept
    : impl_(std::move(other.impl_)), ef_search_(other.ef_search_) {}

IndexHNSW& IndexHNSW::operator=(IndexHNSW&& other) noexcept {
    impl_ = std::move(other.impl_);
    ef_search_ = other.ef_search_;
    return *this;
}

IndexHNSW::~IndexHNSW() = default;

Expected<IndexHNSW> IndexHNSW::create(uint32_t dimension,
                                       const HNSWParams& params,
                                       const IndexOptions& options) {
    if (dimension == 0) {
        return make_unexpected(ErrorCode::InvalidParameter, "Dimension must be positive");
    }

    if (params.M == 0) {
        return make_unexpected(ErrorCode::InvalidParameter, "M must be positive");
    }

    IndexHNSW index;
    index.impl_ = std::make_unique<Impl>();
    index.impl_->dimension = dimension;
    index.impl_->params = params;
    index.impl_->metric = options.metric;

    if (index.impl_->params.ml_factor <= 0.0f) {
        index.impl_->params.ml_factor = 1.0f / std::log(static_cast<float>(params.M));
    }

    index.impl_->data.reserve(1024 * dimension);
    index.impl_->nodes.reserve(1024);
    index.impl_->id_mapping.reserve(1024);

    return index;
}

Expected<IndexHNSW> IndexHNSW::create_gpu(Context& ctx, uint32_t dimension,
                                          const HNSWParams& params,
                                          const IndexOptions& options,
                                          const HNSWGPUOptions& gpu_options) {
    auto base_result = create(dimension, params, options);
    if (!base_result) {
        return base_result;
    }
    
    IndexHNSW index = std::move(*base_result);
    index.impl_->ctx = &ctx;
    index.impl_->gpu_options = gpu_options;
    
    PipelineDesc dist_desc;
    dist_desc.shader_name = "distance_l2";
    dist_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},
        {1, DescriptorBinding::StorageBuffer},
        {2, DescriptorBinding::StorageBuffer}
    };
    dist_desc.push_constant_size = 16;
    
    auto pipeline_result = Pipeline::create(ctx, dist_desc);
    if (!pipeline_result) {
        return index;
    }
    index.impl_->gpu_distance_pipeline = std::move(*pipeline_result);
    
    auto desc_result = DescriptorSet::create(ctx, index.impl_->gpu_distance_pipeline);
    if (!desc_result) {
        return index;
    }
    index.impl_->gpu_distance_desc_set = std::move(*desc_result);
    index.impl_->gpu_initialized = true;
    
    return index;
}

bool IndexHNSW::gpu_enabled() const {
    return impl_ && impl_->gpu_initialized && impl_->gpu_options.enable;
}

uint32_t IndexHNSW::dimension() const {
    return impl_ ? impl_->dimension : 0;
}

uint64_t IndexHNSW::size() const {
    return impl_ ? impl_->n_vectors : 0;
}

IndexStats IndexHNSW::stats() const {
    IndexStats s{};
    if (impl_) {
        s.n_vectors = impl_->n_vectors;
        s.dimension = impl_->dimension;
        s.is_trained = true;

        size_t neighbor_mem = 0;
        for (const auto& node : impl_->nodes) {
            for (const auto& layer : node.neighbors) {
                neighbor_mem += layer.capacity() * sizeof(uint32_t);
            }
        }
        s.memory_used = impl_->data.size() * sizeof(float) +
                        impl_->id_mapping.capacity() * sizeof(VectorId) +
                        neighbor_mem;
    }
    return s;
}

Expected<void> IndexHNSW::add(std::span<const float> data, uint64_t n,
                               std::span<const VectorId> ids) {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) {
        return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");
    }

    if (n == 0) return {};

    std::unique_lock lock(impl_->mutex);

    uint64_t start_id = impl_->n_vectors;
    
    impl_->data.reserve(impl_->data.size() + n * impl_->dimension);
    impl_->nodes.reserve(impl_->nodes.size() + n);
    impl_->id_mapping.reserve(impl_->id_mapping.size() + n);

    const bool auto_ids = ids.empty();
    std::vector<uint32_t> new_node_ids(n, INVALID_NODE);
    std::vector<uint32_t> new_levels(n);
    uint64_t inserted = 0;
    if (auto_ids) {
        const size_t dim = impl_->dimension;
        const size_t old_data_size = impl_->data.size();
        impl_->data.resize(old_data_size + static_cast<size_t>(n) * dim);
        std::memcpy(impl_->data.data() + old_data_size,
                    data.data(),
                    static_cast<size_t>(n) * dim * sizeof(float));

        const size_t old_id_size = impl_->id_mapping.size();
        impl_->id_mapping.resize(old_id_size + static_cast<size_t>(n));
        for (uint64_t i = 0; i < n; ++i) {
            impl_->id_mapping[old_id_size + static_cast<size_t>(i)] =
                static_cast<VectorId>(start_id + i);
        }

        const size_t old_node_size = impl_->nodes.size();
        impl_->nodes.resize(old_node_size + static_cast<size_t>(n));
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t level = impl_->random_level();
            level = std::min(level, MAX_LAYERS - 1);
            new_levels[i] = level;

            const uint32_t node_id = static_cast<uint32_t>(start_id + i);
            new_node_ids[i] = node_id;

            Node& node = impl_->nodes[static_cast<size_t>(node_id)];
            node.id = node_id;
            node.level = level;
            node.neighbors.clear();
            node.neighbors.resize(level + 1);
            for (uint32_t l = 0; l <= level; ++l) {
                const uint32_t max_conn = (l == 0) ? impl_->params.M * 2 : impl_->params.M;
                node.neighbors[l].reserve(max_conn * 4);
            }
        }

        impl_->n_vectors += n;
        inserted = n;
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            VectorId id = ids[i];

            if (impl_->id_set.count(id)) {
                continue;
            }

            const float* vec = data.data() + i * impl_->dimension;

            impl_->data.insert(impl_->data.end(), vec, vec + impl_->dimension);
            impl_->id_mapping.push_back(id);
            impl_->id_set.insert(id);

            uint32_t level = impl_->random_level();
            level = std::min(level, MAX_LAYERS - 1);
            new_levels[i] = level;

            uint32_t node_id = static_cast<uint32_t>(impl_->n_vectors);
            new_node_ids[i] = node_id;

            Node node;
            node.id = node_id;
            node.level = level;
            node.neighbors.resize(level + 1);
            for (uint32_t l = 0; l <= level; ++l) {
                const uint32_t max_conn = (l == 0) ? impl_->params.M * 2 : impl_->params.M;
                node.neighbors[l].reserve(max_conn * 4);
            }
            impl_->nodes.push_back(std::move(node));
            impl_->n_vectors++;
            inserted++;
        }
    }

    if (inserted == 0) {
        return {};
    }

    if (impl_->entry_point == INVALID_NODE && impl_->n_vectors > 0) {
        uint32_t max_lvl = 0;
        uint32_t max_node = INVALID_NODE;
        for (size_t i = 0; i < new_levels.size(); ++i) {
            if (new_node_ids[i] == INVALID_NODE) {
                continue;
            }
            if (max_node == INVALID_NODE || new_levels[i] > max_lvl) {
                max_lvl = new_levels[i];
                max_node = new_node_ids[i];
            }
        }
        if (max_node != INVALID_NODE) {
            impl_->entry_point = max_node;
            impl_->max_level = max_lvl;
        }
    }

    const uint32_t entry_point = impl_->entry_point;
    const int max_search_level = static_cast<int>(impl_->max_level);
    const uint32_t target_threads = static_cast<uint32_t>(std::min<uint64_t>(inserted, n));
    uint32_t num_threads = std::min(std::thread::hardware_concurrency(), target_threads);
#ifdef _OPENMP
    num_threads = std::min(num_threads, static_cast<uint32_t>(std::max(1, omp_get_max_threads())));
#endif
    if (num_threads == 0) num_threads = 1;

    if (entry_point != INVALID_NODE && inserted > 1) {
        BuildLockArray build_locks;

        auto stripe_idx = [](uint32_t node_id) -> size_t {
            return static_cast<size_t>(node_id) & (BUILD_LOCK_STRIPES - 1);
        };

        auto connect_node_parallel = [&](uint32_t node_id, uint32_t level) {
            if (node_id == INVALID_NODE || node_id == entry_point) {
                return;
            }

            const float* vec = impl_->get_vector(node_id);
            std::vector<uint32_t> ep_set = {entry_point};

            for (int lc = max_search_level; lc > static_cast<int>(level); --lc) {
                auto results = impl_->search_layer(vec, static_cast<uint32_t>(lc), 1, ep_set, &build_locks);
                if (!results.empty()) {
                    ep_set = {results[0].second};
                }
            }

            struct LayerSelection {
                uint32_t layer = 0;
                uint32_t max_conn = 0;
                std::vector<uint32_t> neighbors;
            };
            std::vector<LayerSelection> selections;
            selections.reserve(static_cast<size_t>(std::min(static_cast<int>(level), max_search_level) + 1));

            for (int lc = std::min(static_cast<int>(level), max_search_level); lc >= 0; --lc) {
                std::vector<std::pair<float, uint32_t>> candidates;
                const uint32_t max_conn = (lc == 0) ? impl_->params.M * 2 : impl_->params.M;

                auto results = impl_->search_layer(vec,
                                                   static_cast<uint32_t>(lc),
                                                   impl_->params.ef_construction,
                                                   ep_set,
                                                   &build_locks);
                candidates.reserve(results.size());
                for (const auto& r : results) {
                    candidates.push_back(r);
                }
                impl_->select_neighbors_simple(candidates, max_conn);

                LayerSelection sel;
                sel.layer = static_cast<uint32_t>(lc);
                sel.max_conn = max_conn;
                sel.neighbors.reserve(candidates.size());
                for (const auto& [_, n_id] : candidates) {
                    sel.neighbors.push_back(n_id);
                }
                if (!sel.neighbors.empty()) {
                    ep_set = sel.neighbors;
                }
                selections.push_back(std::move(sel));
            }

            for (const auto& sel : selections) {
                const size_t node_stripe = stripe_idx(node_id);
                {
                    std::unique_lock<std::shared_mutex> guard(build_locks[node_stripe]);
                    auto& node_neighbors = impl_->nodes[node_id].neighbors[sel.layer];
                    node_neighbors.reserve(node_neighbors.size() + sel.neighbors.size());
                    for (uint32_t n_id : sel.neighbors) {
                        node_neighbors.push_back(n_id);
                    }
                }

                for (uint32_t n_id : sel.neighbors) {
                    const size_t neighbor_stripe = stripe_idx(n_id);
                    std::unique_lock<std::shared_mutex> guard(build_locks[neighbor_stripe]);
                    Node& neighbor_node = impl_->nodes[n_id];
                    if (sel.layer >= static_cast<uint32_t>(neighbor_node.neighbors.size())) {
                        neighbor_node.neighbors.resize(static_cast<size_t>(sel.layer) + 1);
                    }
                    auto& back_neighbors = neighbor_node.neighbors[sel.layer];
                    back_neighbors.push_back(node_id);

                    // Keep degree bounded but avoid expensive prune every append.
                    if (back_neighbors.size() > (static_cast<size_t>(sel.max_conn) * 2)) {
                        impl_->shrink_connections(n_id, sel.layer, sel.max_conn);
                    }
                }
            }
        };

        if (inserted < 100 || num_threads == 1) {
            for (uint64_t i = 0; i < n; ++i) {
                if (new_node_ids[i] == INVALID_NODE) {
                    continue;
                }
                connect_node_parallel(new_node_ids[i], new_levels[i]);
            }
        }
#ifdef _OPENMP
        else {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
            for (int64_t idx = 0; idx < static_cast<int64_t>(n); ++idx) {
                const uint64_t uidx = static_cast<uint64_t>(idx);
                if (new_node_ids[uidx] == INVALID_NODE) {
                    continue;
                }
                connect_node_parallel(new_node_ids[uidx], new_levels[uidx]);
            }
        }
#else
        else {
            std::atomic<uint64_t> next_idx{0};

            auto worker = [&]() {
                while (true) {
                    const uint64_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= n) break;
                    if (new_node_ids[idx] == INVALID_NODE) {
                        continue;
                    }
                    connect_node_parallel(new_node_ids[idx], new_levels[idx]);
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(num_threads - 1);
            for (uint32_t t = 0; t < num_threads - 1; ++t) {
                threads.emplace_back(worker);
            }
            worker();

            for (auto& t : threads) {
                t.join();
            }
        }
#endif
    }

    // Update entry point after connectivity has been established for the batch.
    for (uint64_t i = 0; i < n; ++i) {
        if (new_node_ids[i] == INVALID_NODE) {
            continue;
        }
        if (new_levels[i] > impl_->max_level) {
            impl_->max_level = new_levels[i];
            impl_->entry_point = new_node_ids[i];
        }
    }

    lock.unlock();

    return {};
}

void IndexHNSW::connect_node(uint32_t node_id, uint32_t level) {
    if (!impl_ || impl_->entry_point == INVALID_NODE || node_id == impl_->entry_point) return;
    
    const float* vec = impl_->get_vector(node_id);
    
    std::vector<uint32_t> ep_set = {impl_->entry_point};
    
    int max_search_level = static_cast<int>(impl_->max_level);
    
    {
        std::shared_lock read_lock(impl_->mutex);
        for (int lc = max_search_level; lc > static_cast<int>(level); --lc) {
            auto results = impl_->search_layer(vec, static_cast<uint32_t>(lc), 1, ep_set);
            if (!results.empty()) {
                ep_set = {results[0].second};
            }
        }
    }
    
    struct LayerSelection {
        uint32_t layer = 0;
        uint32_t max_conn = 0;
        std::vector<uint32_t> neighbors;
    };
    std::vector<LayerSelection> selections;
    selections.reserve(static_cast<size_t>(std::min(static_cast<int>(level), max_search_level) + 1));

    for (int lc = std::min(static_cast<int>(level), max_search_level); lc >= 0; --lc) {
        std::vector<std::pair<float, uint32_t>> candidates;
        uint32_t max_conn = (lc == 0) ? impl_->params.M * 2 : impl_->params.M;

        {
            std::shared_lock read_lock(impl_->mutex);
            auto results = impl_->search_layer(vec, static_cast<uint32_t>(lc), impl_->params.ef_construction, ep_set);
            candidates.reserve(results.size());
            for (const auto& r : results) {
                candidates.push_back(r);
            }
            impl_->select_neighbors_simple(candidates, max_conn);
        }

        LayerSelection sel;
        sel.layer = static_cast<uint32_t>(lc);
        sel.max_conn = max_conn;
        sel.neighbors.reserve(candidates.size());
        for (const auto& [_, n_id] : candidates) {
            sel.neighbors.push_back(n_id);
        }
        if (!sel.neighbors.empty()) {
            ep_set = sel.neighbors;
        }
        selections.push_back(std::move(sel));
    }

    {
        std::unique_lock write_lock(impl_->mutex);
        for (const auto& sel : selections) {
            auto& node_neighbors = impl_->nodes[node_id].neighbors[sel.layer];
            node_neighbors.reserve(node_neighbors.size() + sel.neighbors.size());
            for (uint32_t n_id : sel.neighbors) {
                node_neighbors.push_back(n_id);

                Node& neighbor_node = impl_->nodes[n_id];
                if (sel.layer >= static_cast<uint32_t>(neighbor_node.neighbors.size())) {
                    neighbor_node.neighbors.resize(static_cast<size_t>(sel.layer) + 1);
                }
                auto& back_neighbors = neighbor_node.neighbors[sel.layer];
                back_neighbors.push_back(node_id);

                // Defer pruning until adjacency grows well past target degree.
                if (back_neighbors.size() > (static_cast<size_t>(sel.max_conn) * 2)) {
                    impl_->shrink_connections(n_id, sel.layer, sel.max_conn);
                }
            }
        }
    }
}

Expected<SearchResults> IndexHNSW::search(Vector query, uint32_t k) {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    if (query.size() != impl_->dimension) {
        return make_unexpected(ErrorCode::InvalidDimension, "Query dimension mismatch");
    }

    SearchResults results(1, k);

    if (impl_->n_vectors == 0 || impl_->entry_point == INVALID_NODE) {
        return results;
    }

    std::shared_lock lock(impl_->mutex);

    std::vector<uint32_t> ep_set = {impl_->entry_point};

    for (int lc = static_cast<int>(impl_->max_level); lc > 0; --lc) {
        auto layer_results = impl_->search_layer(query.data(), static_cast<uint32_t>(lc), 1, ep_set);
        if (!layer_results.empty()) {
            ep_set = {layer_results[0].second};
        }
    }

    uint32_t ef = std::max(ef_search_, k);
    auto final_results = impl_->search_layer(query.data(), 0, ef, ep_set);

    uint32_t actual_k = std::min(k, static_cast<uint32_t>(final_results.size()));
    for (uint32_t i = 0; i < actual_k; ++i) {
        results.results[i].distance = final_results[i].first;
        results.results[i].id = impl_->id_mapping[final_results[i].second];
    }

    return results;
}

Expected<SearchResults> IndexHNSW::search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    if (impl_->n_vectors == 0) {
        return SearchResults(n_queries, k);
    }
    
    if (gpu_enabled() && n_queries >= impl_->gpu_options.batch_threshold) {
        return search_batch_gpu(queries, n_queries, k);
    }

    SearchResults results(n_queries, k);
    
    uint32_t num_threads = std::min(std::thread::hardware_concurrency(), 
                                    static_cast<unsigned int>(n_queries));
    if (num_threads == 0) num_threads = 1;
    
    if (n_queries <= 4 || num_threads == 1) {
        std::shared_lock lock(impl_->mutex);
        for (uint64_t q = 0; q < n_queries; ++q) {
            search_single_locked(queries.data() + q * impl_->dimension, k, 
                                 results.results.data() + q * k);
        }
        return results;
    }

    std::shared_lock lock(impl_->mutex);
    std::atomic<uint64_t> next_query{0};
    
    auto worker = [&]() {
        while (true) {
            uint64_t q = next_query.fetch_add(1);
            if (q >= n_queries) break;
            
            search_single_locked(queries.data() + q * impl_->dimension, k, 
                                 results.results.data() + q * k);
        }
    };
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);
    for (uint32_t t = 0; t < num_threads - 1; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    
    for (auto& t : threads) {
        t.join();
    }

    return results;
}

void IndexHNSW::search_single_locked(const float* query, uint32_t k, SearchResult* out) {
    std::vector<uint32_t> ep_set = {impl_->entry_point};

    for (int lc = static_cast<int>(impl_->max_level); lc > 0; --lc) {
        auto layer_results = impl_->search_layer(query, static_cast<uint32_t>(lc), 1, ep_set);
        if (!layer_results.empty()) {
            ep_set = {layer_results[0].second};
        }
    }

    uint32_t ef = std::max(ef_search_, k);
    auto final_results = impl_->search_layer(query, 0, ef, ep_set);

    uint32_t actual_k = std::min(k, static_cast<uint32_t>(final_results.size()));
    for (uint32_t i = 0; i < actual_k; ++i) {
        out[i].distance = final_results[i].first;
        out[i].id = impl_->id_mapping[final_results[i].second];
    }
}

Expected<SearchResults> IndexHNSW::search_batch_gpu(std::span<const float> queries,
                                                    uint64_t n_queries, uint32_t k) {
    if (!impl_->ctx || !impl_->gpu_initialized) {
        return make_unexpected(ErrorCode::OperationFailed, "GPU not initialized");
    }
    
    SearchResults results(n_queries, k);
    
    uint64_t data_size = impl_->n_vectors * impl_->dimension * sizeof(float);
    if (!impl_->gpu_data_buffer.valid() || impl_->gpu_data_buffer.size() < data_size) {
        BufferDesc data_desc = {};
        data_desc.size = data_size;
        data_desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        data_desc.memory_type = MemoryType::DeviceLocal;
        
        auto buf = Buffer::create(*impl_->ctx, data_desc);
        if (!buf) {
            return make_unexpected(buf.error().code(), "Failed to create GPU data buffer");
        }
        impl_->gpu_data_buffer = std::move(*buf);
        
        std::vector<uint8_t> bytes(impl_->data.size() * sizeof(float));
        std::memcpy(bytes.data(), impl_->data.data(), bytes.size());
        auto upload_result = impl_->gpu_data_buffer.upload(bytes);
        if (!upload_result) {
            return make_unexpected(upload_result.error().code(), "Failed to upload data to GPU");
        }
    }
    
    uint32_t num_threads = std::min(std::thread::hardware_concurrency(), 
                                    static_cast<unsigned int>(n_queries));
    if (num_threads == 0) num_threads = 1;
    
    std::shared_lock lock(impl_->mutex);
    std::atomic<uint64_t> next_query{0};
    
    auto worker = [&]() {
        while (true) {
            uint64_t q = next_query.fetch_add(1);
            if (q >= n_queries) break;
            
            search_single_locked(queries.data() + q * impl_->dimension, k, 
                                 results.results.data() + q * k);
        }
    };
    
    if (num_threads > 1) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads - 1);
        for (uint32_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker);
        }
        worker();
        for (auto& t : threads) {
            t.join();
        }
    } else {
        for (uint64_t q = 0; q < n_queries; ++q) {
            search_single_locked(queries.data() + q * impl_->dimension, k, 
                                 results.results.data() + q * k);
        }
    }
    
    return results;
}

Expected<void> IndexHNSW::save(const std::filesystem::path& path) const {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    std::shared_lock lock(impl_->mutex);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return make_unexpected(ErrorCode::WriteFailed, "Failed to open file for writing");
    }

    uint32_t magic = 0x484E5357;
    uint32_t version = 1;

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&impl_->dimension), sizeof(impl_->dimension));
    out.write(reinterpret_cast<const char*>(&impl_->n_vectors), sizeof(impl_->n_vectors));
    out.write(reinterpret_cast<const char*>(&impl_->metric), sizeof(impl_->metric));
    out.write(reinterpret_cast<const char*>(&impl_->params.M), sizeof(impl_->params.M));
    out.write(reinterpret_cast<const char*>(&impl_->params.ef_construction), sizeof(impl_->params.ef_construction));
    out.write(reinterpret_cast<const char*>(&impl_->entry_point), sizeof(impl_->entry_point));
    out.write(reinterpret_cast<const char*>(&impl_->max_level), sizeof(impl_->max_level));

    out.write(reinterpret_cast<const char*>(impl_->data.data()),
              impl_->data.size() * sizeof(float));

    uint64_t id_count = impl_->id_mapping.size();
    out.write(reinterpret_cast<const char*>(&id_count), sizeof(id_count));
    out.write(reinterpret_cast<const char*>(impl_->id_mapping.data()),
              id_count * sizeof(VectorId));

    for (const auto& node : impl_->nodes) {
        uint32_t level = node.level;
        out.write(reinterpret_cast<const char*>(&level), sizeof(level));

        for (uint32_t l = 0; l <= level; ++l) {
            uint32_t n_neighbors = static_cast<uint32_t>(node.neighbors[l].size());
            out.write(reinterpret_cast<const char*>(&n_neighbors), sizeof(n_neighbors));
            if (n_neighbors > 0) {
                out.write(reinterpret_cast<const char*>(node.neighbors[l].data()),
                          n_neighbors * sizeof(uint32_t));
            }
        }
    }

    if (!out) {
        return make_unexpected(ErrorCode::WriteFailed, "Failed to write index data");
    }

    return {};
}

Expected<void> IndexHNSW::load(const std::filesystem::path& path) {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }

    std::unique_lock lock(impl_->mutex);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return make_unexpected(ErrorCode::FileNotFound, "Failed to open file for reading");
    }

    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x484E5357) {
        return make_unexpected(ErrorCode::InvalidFileFormat, "Invalid file format");
    }

    in.read(reinterpret_cast<char*>(&impl_->dimension), sizeof(impl_->dimension));
    in.read(reinterpret_cast<char*>(&impl_->n_vectors), sizeof(impl_->n_vectors));
    in.read(reinterpret_cast<char*>(&impl_->metric), sizeof(impl_->metric));
    in.read(reinterpret_cast<char*>(&impl_->params.M), sizeof(impl_->params.M));
    in.read(reinterpret_cast<char*>(&impl_->params.ef_construction), sizeof(impl_->params.ef_construction));
    in.read(reinterpret_cast<char*>(&impl_->entry_point), sizeof(impl_->entry_point));
    in.read(reinterpret_cast<char*>(&impl_->max_level), sizeof(impl_->max_level));

    impl_->data.resize(static_cast<size_t>(impl_->n_vectors) * impl_->dimension);
    in.read(reinterpret_cast<char*>(impl_->data.data()),
            impl_->data.size() * sizeof(float));

    uint64_t id_count;
    in.read(reinterpret_cast<char*>(&id_count), sizeof(id_count));
    impl_->id_mapping.resize(id_count);
    in.read(reinterpret_cast<char*>(impl_->id_mapping.data()),
            id_count * sizeof(VectorId));

    impl_->id_set.clear();
    for (VectorId id : impl_->id_mapping) {
        impl_->id_set.insert(id);
    }

    impl_->nodes.resize(impl_->n_vectors);
    for (uint64_t i = 0; i < impl_->n_vectors; ++i) {
        Node& node = impl_->nodes[i];
        node.id = static_cast<uint32_t>(i);

        uint32_t level;
        in.read(reinterpret_cast<char*>(&level), sizeof(level));
        node.level = level;
        node.neighbors.resize(level + 1);

        for (uint32_t l = 0; l <= level; ++l) {
            uint32_t n_neighbors;
            in.read(reinterpret_cast<char*>(&n_neighbors), sizeof(n_neighbors));
            node.neighbors[l].resize(n_neighbors);
            if (n_neighbors > 0) {
                in.read(reinterpret_cast<char*>(node.neighbors[l].data()),
                        n_neighbors * sizeof(uint32_t));
            }
        }
    }

    if (!in) {
        return make_unexpected(ErrorCode::ReadFailed, "Failed to read index data");
    }

    return {};
}

void IndexHNSW::reset() {
    if (impl_) {
        std::unique_lock lock(impl_->mutex);
        impl_->n_vectors = 0;
        impl_->data.clear();
        impl_->nodes.clear();
        impl_->id_mapping.clear();
        impl_->id_set.clear();
        impl_->entry_point = INVALID_NODE;
        impl_->max_level = 0;
    }
}

} // namespace cw
