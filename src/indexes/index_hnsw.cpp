#include <catwhisper/index_hnsw.hpp>
#include <catwhisper/distance.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_set>
#include <random>
#include <shared_mutex>

namespace cw {

static constexpr uint32_t MAX_LAYERS = 64;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;

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

    float distance(const float* a, const float* b) const {
        if (metric == Metric::IP) {
            float ip = 0.0f;
            for (uint32_t i = 0; i < dimension; ++i) {
                ip += a[i] * b[i];
            }
            return -ip;
        } else {
            float dist = 0.0f;
            for (uint32_t i = 0; i < dimension; ++i) {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }
            return dist;
        }
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
                 const std::vector<uint32_t>& entry_points) {
        std::unordered_set<uint32_t> visited;
        visited.reserve(ef * 4);

        MinHeap candidates;  // min-heap: smallest dist on top
        MaxHeap results;     // max-heap: largest dist on top

        for (uint32_t ep : entry_points) {
            float dist = distance(query, get_vector(ep));
            visited.insert(ep);
            candidates.emplace(dist, ep);
            results.emplace(dist, ep);
        }

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();

            float f_dist = results.top().first;

            // Don't terminate early - explore all reachable candidates for better recall
            // Original: if (c_dist > f_dist && results.size() >= ef) break;

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

    // Auto-compute ml_factor if not set (optimal: 1/ln(M))
    if (index.impl_->params.ml_factor <= 0.0f) {
        index.impl_->params.ml_factor = 1.0f / std::log(static_cast<float>(params.M));
    }

    index.impl_->data.reserve(1024 * dimension);
    index.impl_->nodes.reserve(1024);
    index.impl_->id_mapping.reserve(1024);

    return index;
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

    uint64_t base_id = impl_->n_vectors;
    for (uint64_t i = 0; i < n; ++i) {
        VectorId id = ids.empty() ? static_cast<VectorId>(base_id + i) : ids[i];
        auto result = impl_->insert_node(data.data() + i * impl_->dimension, id);
        if (!result) {
            return result;
        }
    }

    return {};
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

    SearchResults results(n_queries, k);

    for (uint64_t q = 0; q < n_queries; ++q) {
        std::vector<float> query(queries.begin() + q * impl_->dimension,
                                  queries.begin() + (q + 1) * impl_->dimension);
        auto result = search(query, k);
        if (!result) {
            return result;
        }

        for (uint32_t i = 0; i < k; ++i) {
            results.results[q * k + i] = result->results[i];
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
