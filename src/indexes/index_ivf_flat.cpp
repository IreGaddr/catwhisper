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
#include <iostream>

namespace cw {

// Float32 to float16 conversion
static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 31) & 1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent > 15) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    if (exponent < -14) {
        return static_cast<uint16_t>(sign << 15);
    }

    int32_t new_exp = exponent + 15;
    uint32_t new_mant = mantissa >> 13;

    return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | new_mant);
}

// K-means clustering (CPU - done once during training)
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
            if (update_centroids(data, n_samples, assignments.data(), centroids.data())) break;
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
                if (dist < min_distances[i]) min_distances[i] = dist;
                total_dist += min_distances[i];
            }

            std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
            double threshold = prob_dist(rng);
            double cumulative = 0.0;
            uint64_t next_idx = 0;
            for (uint64_t i = 0; i < n_samples; ++i) {
                cumulative += min_distances[i];
                if (cumulative >= threshold) { next_idx = i; break; }
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

    void assign_clusters(const float* data, uint64_t n_samples, const float* centroids, uint32_t* assignments) {
        for (uint64_t i = 0; i < n_samples; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            uint32_t best = 0;
            for (uint32_t c = 0; c < n_clusters_; ++c) {
                double dist = compute_distance(data + i * dimension_, centroids + c * dimension_);
                if (dist < min_dist) { min_dist = dist; best = c; }
            }
            assignments[i] = best;
        }
    }

    bool update_centroids(const float* data, uint64_t n_samples, const uint32_t* assignments, float* centroids) {
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
                    if (std::abs(new_val - centroids[c * dimension_ + d]) > 1e-6f) converged = false;
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
    uint64_t capacity = 0;
    Metric metric = Metric::L2;
    bool use_fp16 = true;

    IVFParams params;
    uint32_t actual_nlist = 0;

    // CPU-side data for building
    std::vector<float> centroids;
    std::vector<uint32_t> cluster_offsets;  // nlist + 1, offsets into data buffer
    std::vector<std::vector<float>> invlists_data;  // Temporary storage during building
    std::vector<std::vector<uint64_t>> invlists_ids;
    std::vector<uint64_t> flat_ids;  // Cached: IDs in cluster-major order
    std::vector<uint32_t> cluster_soa_offsets;  // SoA start offsets per cluster (in fp16 units)

    // GPU buffers
    Buffer centroids_buffer;      // nlist * dimension fp32
    Buffer data_buffer;           // vectors in cluster-major fp16
    Buffer offsets_buffer;        // nlist + 1 uint32
    Buffer ids_buffer;            // vector IDs uint64

    // Search buffers
    Buffer query_buffer;          // dimension fp16
    Buffer cluster_info_buffer;   // nprobe * sizeof(uvec4)
    Buffer result_dists_buffer;   // nprobe * k floats
    Buffer result_idxs_buffer;    // nprobe * k uints

    // Pipeline
    Pipeline search_pipeline;
    DescriptorSet search_desc_set;
    CommandBuffer search_cmd;
    bool search_cmd_valid = false;
    uint32_t search_cmd_k = 0;
    uint32_t search_cmd_nprobe = 0;

    // Assign pipeline
    Pipeline assign_pipeline;
    DescriptorSet assign_desc_set;

    // Assign buffers (reused across add() calls)
    Buffer assign_vectors_buf;    // input vectors (fp16)
    Buffer assign_output_buf;     // cluster assignments (uint32)

    bool is_trained = false;
    bool gpu_dirty = true;  // Need to rebuild GPU buffers
};

IndexIVFFlat::IndexIVFFlat(IndexIVFFlat&& other) noexcept : impl_(std::move(other.impl_)) {}
IndexIVFFlat& IndexIVFFlat::operator=(IndexIVFFlat&& other) noexcept { impl_ = std::move(other.impl_); return *this; }
IndexIVFFlat::~IndexIVFFlat() = default;

Expected<IndexIVFFlat> IndexIVFFlat::create(Context& ctx, uint32_t dimension, const IVFParams& params, const IndexOptions& options) {
    IndexIVFFlat index;
    index.impl_ = std::make_unique<Impl>();
    index.impl_->ctx = &ctx;
    index.impl_->dimension = dimension;
    index.impl_->metric = options.metric;
    index.impl_->use_fp16 = options.use_fp16;
    index.impl_->params = params;

    index.impl_->invlists_data.resize(params.nlist);
    index.impl_->invlists_ids.resize(params.nlist);
    index.impl_->cluster_offsets.resize(params.nlist + 1, 0);

    // Initialize pipeline
    auto pipeline_result = index.init_pipelines();
    if (!pipeline_result) {
        return make_unexpected(pipeline_result.error().code(), pipeline_result.error().message());
    }

    return index;
}

Expected<void> IndexIVFFlat::init_pipelines() {
    // Search pipeline
    {
        PipelineDesc desc;
        desc.shader_name = "ivf_distance";
        desc.bindings = {
            {0, DescriptorBinding::StorageBuffer},  // cluster_data
            {1, DescriptorBinding::StorageBuffer},  // cluster_info
            {2, DescriptorBinding::StorageBuffer},  // query
            {3, DescriptorBinding::StorageBuffer},  // out_distances
            {4, DescriptorBinding::StorageBuffer},  // out_indices
        };
        desc.push_constant_size = 16;  // dimension, k, metric, pad

        auto result = Pipeline::create(*impl_->ctx, desc);
        if (!result) {
            return make_unexpected(result.error().code(), "Failed to create IVF search pipeline: " + result.error().message());
        }
        impl_->search_pipeline = std::move(*result);

        auto desc_result = DescriptorSet::create(*impl_->ctx, impl_->search_pipeline);
        if (!desc_result) {
            return make_unexpected(desc_result.error().code(), "Failed to create IVF descriptor set");
        }
        impl_->search_desc_set = std::move(*desc_result);
    }

    // Assign pipeline
    {
        PipelineDesc desc;
        desc.shader_name = "assign_clusters";
        desc.bindings = {
            {0, DescriptorBinding::StorageBuffer},  // centroids
            {1, DescriptorBinding::StorageBuffer},  // vectors
            {2, DescriptorBinding::StorageBuffer},  // cluster_ids output
        };
        desc.push_constant_size = 20;  // n_clusters, dimension, n_vectors, metric, padding

        auto result = Pipeline::create(*impl_->ctx, desc);
        if (!result) {
            return make_unexpected(result.error().code(), "Failed to create assign pipeline: " + result.error().message());
        }
        impl_->assign_pipeline = std::move(*result);

        auto desc_result = DescriptorSet::create(*impl_->ctx, impl_->assign_pipeline);
        if (!desc_result) {
            return make_unexpected(desc_result.error().code(), "Failed to create assign descriptor set");
        }
        impl_->assign_desc_set = std::move(*desc_result);
    }

    return {};
}

uint32_t IndexIVFFlat::dimension() const { return impl_ ? impl_->dimension : 0; }
uint64_t IndexIVFFlat::size() const { return impl_ ? impl_->n_vectors : 0; }
bool IndexIVFFlat::is_trained() const { return impl_ ? impl_->is_trained : false; }
uint32_t IndexIVFFlat::nlist() const { return impl_ ? impl_->actual_nlist : 0; }
uint32_t IndexIVFFlat::nprobe() const { return impl_ ? impl_->params.nprobe : 0; }
void IndexIVFFlat::set_nprobe(uint32_t nprobe) { if (impl_) impl_->params.nprobe = std::min(nprobe, impl_->actual_nlist); }

IndexStats IndexIVFFlat::stats() const {
    IndexStats s{};
    if (impl_) {
        s.n_vectors = impl_->n_vectors;
        s.dimension = impl_->dimension;
        s.is_trained = impl_->is_trained;
        s.gpu_memory_used = impl_->data_buffer.size() + impl_->centroids_buffer.size() + impl_->ids_buffer.size();
    }
    return s;
}

Expected<void> IndexIVFFlat::train(std::span<const float> data, uint64_t n) {
    if (!impl_ || !impl_->ctx) return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");

    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");

    // Run K-means
    KMeans kmeans(impl_->params.nlist, impl_->dimension, impl_->params.kmeans_iters, impl_->metric);
    impl_->centroids = kmeans.fit(data.data(), n);
    impl_->actual_nlist = kmeans.nclusters();
    impl_->params.nprobe = std::min(impl_->params.nprobe, impl_->actual_nlist);

    // Upload centroids to GPU
    uint64_t centroids_size = impl_->actual_nlist * impl_->dimension * sizeof(float);
    if (!impl_->centroids_buffer.valid() || impl_->centroids_buffer.size() < centroids_size) {
        BufferDesc desc = {};
        desc.size = centroids_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::DeviceLocal;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create centroids buffer");
        impl_->centroids_buffer = std::move(*buf);
    }

    std::vector<uint8_t> centroid_bytes(centroids_size);
    std::memcpy(centroid_bytes.data(), impl_->centroids.data(), centroids_size);
    auto upload_result = impl_->centroids_buffer.upload(centroid_bytes);
    if (!upload_result) return upload_result;

    // Update cluster offsets buffer
    impl_->cluster_offsets.resize(impl_->actual_nlist + 1, 0);

    impl_->is_trained = true;
    return {};
}

Expected<void> IndexIVFFlat::add(std::span<const float> data, uint64_t n, std::span<const VectorId> ids) {
    if (!impl_ || !impl_->ctx) return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    if (!impl_->is_trained) return make_unexpected(ErrorCode::InvalidParameter, "Index must be trained first");

    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) return make_unexpected(ErrorCode::InvalidParameter, "Data size mismatch");

    if (n == 0) return {};

    // Allocate/reallocate GPU buffers for assignment
    uint64_t vectors_size = n * impl_->dimension * sizeof(uint16_t);
    if (!impl_->assign_vectors_buf.valid() || impl_->assign_vectors_buf.size() < vectors_size) {
        BufferDesc desc = {};
        desc.size = vectors_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::HostCoherent;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create assign vectors buffer");
        impl_->assign_vectors_buf = std::move(*buf);
    }

    uint64_t output_size = n * sizeof(uint32_t);
    if (!impl_->assign_output_buf.valid() || impl_->assign_output_buf.size() < output_size) {
        BufferDesc desc = {};
        desc.size = output_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
        desc.memory_type = MemoryType::HostReadback;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create assign output buffer");
        impl_->assign_output_buf = std::move(*buf);
    }

    // Convert and upload vectors to fp16
    {
        uint16_t* mapped = reinterpret_cast<uint16_t*>(impl_->assign_vectors_buf.mapped());
        for (uint64_t i = 0; i < n * impl_->dimension; ++i) {
            mapped[i] = float_to_half(data[i]);
        }
    }

    // Bind descriptor set
    auto bind0 = impl_->assign_desc_set.bind_buffer(0, impl_->centroids_buffer);
    auto bind1 = impl_->assign_desc_set.bind_buffer(1, impl_->assign_vectors_buf);
    auto bind2 = impl_->assign_desc_set.bind_buffer(2, impl_->assign_output_buf);

    if (!bind0 || !bind1 || !bind2) {
        return make_unexpected(ErrorCode::OperationFailed, "Failed to bind assign descriptor set");
    }

    // Create command buffer and dispatch
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) return make_unexpected(cmd_result.error().code(), cmd_result.error().message());

    CommandBuffer cmd = std::move(*cmd_result);

    struct PushConstants {
        uint32_t n_clusters;
        uint32_t dimension;
        uint32_t n_vectors;
        uint32_t metric;
        uint32_t padding;
    } pc = {
        impl_->actual_nlist,
        impl_->dimension,
        static_cast<uint32_t>(n),
        (impl_->metric == Metric::IP) ? 1u : 0u,
        0u
    };

    cmd.begin();
    cmd.bind_pipeline(impl_->assign_pipeline);
    cmd.bind_descriptor_set(impl_->assign_pipeline, impl_->assign_desc_set);
    cmd.push_constants(impl_->assign_pipeline, &pc, sizeof(pc));
    cmd.dispatch(static_cast<uint32_t>(n));  // One workgroup per vector
    cmd.end();

    auto submit_result = submit_and_wait(*impl_->ctx, cmd);
    if (!submit_result) return make_unexpected(submit_result.error().code(), submit_result.error().message());

    // Read back cluster assignments
    const uint32_t* cluster_ids = reinterpret_cast<const uint32_t*>(impl_->assign_output_buf.mapped());

    // Store vectors in appropriate inverted lists
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t cluster = cluster_ids[i];
        if (cluster >= impl_->actual_nlist) {
            cluster = 0;  // Fallback to first cluster if invalid
        }

        const float* vec = data.data() + i * impl_->dimension;
        impl_->invlists_data[cluster].insert(impl_->invlists_data[cluster].end(), vec, vec + impl_->dimension);

        VectorId external_id = ids.empty() ? (impl_->n_vectors + i) : ids[i];
        impl_->invlists_ids[cluster].push_back(external_id);
    }

    impl_->n_vectors += n;
    impl_->gpu_dirty = true;
    return {};
}

Expected<void> IndexIVFFlat::upload_to_gpu() {
    if (!impl_->gpu_dirty || impl_->n_vectors == 0) return {};

    // Compute cluster offsets
    impl_->cluster_offsets[0] = 0;
    for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
        impl_->cluster_offsets[c + 1] = impl_->cluster_offsets[c] + static_cast<uint32_t>(impl_->invlists_ids[c].size());
    }

    // Build cluster data buffer in SoA format (per-cluster SoA)
    // For cluster c with n vectors: data[c_start + d * n + v] = dim d of vector v
    uint64_t data_size = impl_->n_vectors * impl_->dimension * sizeof(uint16_t);

    if (!impl_->data_buffer.valid() || impl_->data_buffer.size() < data_size) {
        BufferDesc desc = {};
        desc.size = data_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::DeviceLocal;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create data buffer");
        impl_->data_buffer = std::move(*buf);
    }

    // Flatten data in per-cluster SoA format
    std::vector<uint16_t> flat_data(impl_->n_vectors * impl_->dimension);
    uint64_t write_pos = 0;

    // Track byte offsets for each cluster (in fp16 units)
    impl_->cluster_soa_offsets.resize(impl_->actual_nlist);

    for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
        uint32_t cluster_size = impl_->cluster_offsets[c + 1] - impl_->cluster_offsets[c];
        impl_->cluster_soa_offsets[c] = write_pos;  // Offset in fp16 units

        // SoA: for each dimension, write all vectors in this cluster
        for (uint32_t d = 0; d < impl_->dimension; ++d) {
            for (uint32_t v = 0; v < cluster_size; ++v) {
                float val = impl_->invlists_data[c][v * impl_->dimension + d];
                flat_data[write_pos++] = float_to_half(val);
            }
        }
    }

    std::vector<uint8_t> data_bytes(flat_data.size() * sizeof(uint16_t));
    std::memcpy(data_bytes.data(), flat_data.data(), data_bytes.size());
    auto upload_data = impl_->data_buffer.upload(data_bytes);
    if (!upload_data) return upload_data;

    // Upload cluster offsets
    uint64_t offsets_size = (impl_->actual_nlist + 1) * sizeof(uint32_t);
    if (!impl_->offsets_buffer.valid() || impl_->offsets_buffer.size() < offsets_size) {
        BufferDesc desc = {};
        desc.size = offsets_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::DeviceLocal;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create offsets buffer");
        impl_->offsets_buffer = std::move(*buf);
    }

    std::vector<uint8_t> offset_bytes(offsets_size);
    std::memcpy(offset_bytes.data(), impl_->cluster_offsets.data(), offsets_size);
    auto upload_offsets = impl_->offsets_buffer.upload(offset_bytes);
    if (!upload_offsets) return upload_offsets;

    // Upload IDs in cluster-major order
    uint64_t ids_size = impl_->n_vectors * sizeof(uint64_t);
    if (!impl_->ids_buffer.valid() || impl_->ids_buffer.size() < ids_size) {
        BufferDesc desc = {};
        desc.size = ids_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::DeviceLocal;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create IDs buffer");
        impl_->ids_buffer = std::move(*buf);
    }

    // Build and cache flat_ids in cluster-major order
    impl_->flat_ids.resize(impl_->n_vectors);
    uint64_t id_offset = 0;
    for (uint32_t c = 0; c < impl_->actual_nlist; ++c) {
        for (uint64_t id : impl_->invlists_ids[c]) {
            impl_->flat_ids[id_offset++] = id;
        }
    }

    std::vector<uint8_t> id_bytes(ids_size);
    std::memcpy(id_bytes.data(), impl_->flat_ids.data(), ids_size);
    auto upload_ids = impl_->ids_buffer.upload(id_bytes);
    if (!upload_ids) return upload_ids;

    impl_->gpu_dirty = false;
    impl_->search_cmd_valid = false;
    return {};
}

Expected<SearchResults> IndexIVFFlat::search(Vector query, uint32_t k) {
    if (!impl_ || !impl_->ctx) return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    if (query.size() != impl_->dimension) return make_unexpected(ErrorCode::InvalidDimension, "Query dimension mismatch");
    if (impl_->n_vectors == 0) return SearchResults(1, k);

    // Upload data to GPU if dirty
    auto upload_result = upload_to_gpu();
    if (!upload_result) return make_unexpected(upload_result.error().code(), upload_result.error().message());

    uint32_t nprobe = std::min(impl_->params.nprobe, impl_->actual_nlist);

    // Upload query to GPU (fp16)
    if (!impl_->query_buffer.valid() || impl_->query_buffer.size() < impl_->dimension * sizeof(uint16_t)) {
        BufferDesc desc = {};
        desc.size = impl_->dimension * sizeof(uint16_t);
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::HostCoherent;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create query buffer");
        impl_->query_buffer = std::move(*buf);
    }

    {
        uint16_t* mapped = reinterpret_cast<uint16_t*>(impl_->query_buffer.mapped());
        for (uint32_t d = 0; d < impl_->dimension; ++d) {
            mapped[d] = float_to_half(query[d]);
        }
    }

    // CPU: Select top-nprobe centroids
    std::vector<std::pair<float, uint32_t>> centroid_dists;
    centroid_dists.reserve(impl_->actual_nlist);
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
        centroid_dists.emplace_back(dist, c);
    }
    
    // Partial sort to get top-nprobe
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end());
    
    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(nprobe);
    for (uint32_t p = 0; p < nprobe; ++p) {
        selected_clusters.push_back(centroid_dists[p].second);
    }

    // Allocate cluster_info buffer for nprobe clusters (uvec4 per cluster)
    uint64_t cluster_info_size = nprobe * sizeof(uint32_t) * 4;
    if (!impl_->cluster_info_buffer.valid() || impl_->cluster_info_buffer.size() < cluster_info_size) {
        BufferDesc desc = {};
        desc.size = cluster_info_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        desc.memory_type = MemoryType::HostCoherent;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create cluster info buffer");
        impl_->cluster_info_buffer = std::move(*buf);
    }

    // Populate cluster_info buffer (SoA offsets in fp16 units)
    {
        uint32_t* mapped = reinterpret_cast<uint32_t*>(impl_->cluster_info_buffer.mapped());
        for (uint32_t p = 0; p < nprobe; ++p) {
            uint32_t cluster = selected_clusters[p];
            mapped[p * 4 + 0] = impl_->cluster_soa_offsets[cluster];  // SoA offset in fp16 units
            mapped[p * 4 + 1] = impl_->cluster_offsets[cluster + 1] - impl_->cluster_offsets[cluster];  // size
            mapped[p * 4 + 2] = p;  // output_slot
            mapped[p * 4 + 3] = cluster;
        }
    }

    // Allocate result buffers for nprobe * k results
    uint64_t result_size = nprobe * k * sizeof(float);
    if (!impl_->result_dists_buffer.valid() || impl_->result_dists_buffer.size() < result_size) {
        BufferDesc desc = {};
        desc.size = result_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
        desc.memory_type = MemoryType::HostReadback;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create result distances buffer");
        impl_->result_dists_buffer = std::move(*buf);
    }

    result_size = nprobe * k * sizeof(uint32_t);
    if (!impl_->result_idxs_buffer.valid() || impl_->result_idxs_buffer.size() < result_size) {
        BufferDesc desc = {};
        desc.size = result_size;
        desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
        desc.memory_type = MemoryType::HostReadback;
        desc.map_on_create = true;

        auto buf = Buffer::create(*impl_->ctx, desc);
        if (!buf) return make_unexpected(buf.error().code(), "Failed to create result indices buffer");
        impl_->result_idxs_buffer = std::move(*buf);
    }

    // Bind descriptor set (new interface)
    auto bind0 = impl_->search_desc_set.bind_buffer(0, impl_->data_buffer);
    auto bind1 = impl_->search_desc_set.bind_buffer(1, impl_->cluster_info_buffer);
    auto bind2 = impl_->search_desc_set.bind_buffer(2, impl_->query_buffer);
    auto bind3 = impl_->search_desc_set.bind_buffer(3, impl_->result_dists_buffer);
    auto bind4 = impl_->search_desc_set.bind_buffer(4, impl_->result_idxs_buffer);

    if (!bind0 || !bind1 || !bind2 || !bind3 || !bind4) {
        return make_unexpected(ErrorCode::OperationFailed, "Failed to bind descriptor set");
    }

    // ---- Record or reuse command buffer ----
    bool need_rerecord = !impl_->search_cmd_valid ||
                         impl_->search_cmd_k != k ||
                         impl_->search_cmd_nprobe != nprobe;

    if (need_rerecord) {
        if (!impl_->search_cmd.valid()) {
            auto cmd_result = CommandBuffer::create(*impl_->ctx);
            if (!cmd_result) return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
            impl_->search_cmd = std::move(*cmd_result);
        } else {
            impl_->search_cmd.reset();
        }

        struct PushConstants {
            uint32_t dimension;
            uint32_t k;
            uint32_t metric;
            uint32_t pad0;
        } pc = {
            impl_->dimension,
            k,
            (impl_->metric == Metric::IP) ? 1u : 0u,
            0u
        };

        impl_->search_cmd.begin_reusable();
        impl_->search_cmd.bind_pipeline(impl_->search_pipeline);
        impl_->search_cmd.bind_descriptor_set(impl_->search_pipeline, impl_->search_desc_set);
        impl_->search_cmd.push_constants(impl_->search_pipeline, &pc, sizeof(pc));
        impl_->search_cmd.dispatch(nprobe);
        impl_->search_cmd.end();

        impl_->search_cmd_valid = true;
        impl_->search_cmd_k = k;
        impl_->search_cmd_nprobe = nprobe;
    }

    // Submit and wait
    auto submit_result = submit_and_wait(*impl_->ctx, impl_->search_cmd);
    if (!submit_result) return make_unexpected(submit_result.error().code(), submit_result.error().message());

    // Read and merge results from all clusters
    // Shader returns local position within cluster; need to convert to global index
    const float* dists = reinterpret_cast<const float*>(impl_->result_dists_buffer.mapped());
    const uint32_t* idxs = reinterpret_cast<const uint32_t*>(impl_->result_idxs_buffer.mapped());

    // Collect all results from all clusters, converting local pos to global index
    std::vector<std::pair<float, uint32_t>> all_results;
    all_results.reserve(nprobe * k);
    for (uint32_t p = 0; p < nprobe; ++p) {
        uint32_t cluster = selected_clusters[p];
        uint32_t cluster_start = impl_->cluster_offsets[cluster];
        uint32_t cluster_size = impl_->cluster_offsets[cluster + 1] - impl_->cluster_offsets[cluster];
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t local_pos = idxs[p * k + i];
            if (local_pos != 0xFFFFFFFFu && local_pos < cluster_size) {
                uint32_t global_idx = cluster_start + local_pos;
                all_results.emplace_back(dists[p * k + i], global_idx);
            }
        }
    }

    // Sort and take top-k
    std::partial_sort(all_results.begin(), 
                      all_results.begin() + std::min(k, static_cast<uint32_t>(all_results.size())),
                      all_results.end());
    all_results.resize(std::min(k, static_cast<uint32_t>(all_results.size())));

    // Map buffer indices to external IDs
    SearchResults results(1, k);
    for (size_t i = 0; i < all_results.size(); ++i) {
        results.results[i].distance = all_results[i].first;
        results.results[i].id = impl_->flat_ids[all_results[i].second];
    }

    return results;
}

Expected<SearchResults> IndexIVFFlat::search(std::span<const float> queries, uint64_t n_queries, uint32_t k) {
    if (!impl_ || !impl_->ctx) return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    if (impl_->n_vectors == 0) return SearchResults(n_queries, k);

    SearchResults results(n_queries, k);
    for (uint64_t q = 0; q < n_queries; ++q) {
        std::vector<float> query(queries.begin() + q * impl_->dimension, queries.begin() + (q + 1) * impl_->dimension);
        auto single = search(query, k);
        if (!single) return single;
        for (uint32_t i = 0; i < k; ++i) {
            results.results[q * k + i] = single->results[i];
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
        impl_->gpu_dirty = true;
        impl_->search_cmd_valid = false;
        for (auto& list : impl_->invlists_data) list.clear();
        for (auto& list : impl_->invlists_ids) list.clear();
        std::fill(impl_->cluster_offsets.begin(), impl_->cluster_offsets.end(), 0);
    }
}

} // namespace cw
