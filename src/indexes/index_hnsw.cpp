#include <catwhisper/index_hnsw.hpp>
#include <catwhisper/distance.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>

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
#include <thread>
#include <atomic>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

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
    
    Context* ctx = nullptr;
    HNSWGPUOptions gpu_options;
    bool gpu_initialized = false;
    
    Buffer gpu_data_buffer;
    Buffer gpu_query_buffer;
    Buffer gpu_distance_buffer;
    Pipeline gpu_distance_pipeline;
    DescriptorSet gpu_distance_desc_set;

    float distance(const float* a, const float* b) const {
        if (metric == Metric::IP) {
            return -distance::inner_product({a, dimension}, {b, dimension});
        }
        return distance::l2_sqr({a, dimension}, {b, dimension});
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
