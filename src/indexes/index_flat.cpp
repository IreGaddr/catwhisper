#include <catwhisper/index_flat.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>
#include "core/context_impl.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>

namespace cw {

// Helper functions for float16 conversion
static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    if (exponent > 15) {
        uint16_t inf = static_cast<uint16_t>((sign << 15) | 0x7C00);
        return inf;
    }
    if (exponent < -14) {
        return static_cast<uint16_t>(sign << 15);
    }
    
    int32_t new_exp = exponent + 15;
    uint32_t new_mant = mantissa >> 13;
    
    return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | new_mant);
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t bits = sign << 31;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            return f;
        }
    } else if (exponent == 31) {
        uint32_t bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }
    
    uint32_t new_exp = exponent + (127 - 15);
    uint32_t bits = (sign << 31) | (new_exp << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

// TOPK_CHUNK must match the CHUNK constant in topk_heap.comp.
static constexpr uint32_t TOPK_CHUNK = 512u;

struct IndexFlat::Impl {
    Context* ctx = nullptr;
    uint32_t dimension = 0;
    uint64_t n_vectors = 0;
    uint64_t capacity = 0;
    Metric metric = Metric::L2;
    bool use_fp16 = true;

    Buffer data_buffer;
    Buffer ids_buffer;

    Pipeline distance_l2_pipeline;
    Pipeline distance_ip_pipeline;
    Pipeline topk_pipeline;
    DescriptorSet distance_desc_set;
    DescriptorSet topk_desc_set;

    Buffer query_buffer;
    Buffer distance_buffer;
    Buffer result_distances_buffer;
    Buffer result_indices_buffer;

    std::vector<VectorId> id_mapping;
};

IndexFlat::IndexFlat(IndexFlat&& other) noexcept
    : impl_(std::move(other.impl_)) {}

IndexFlat& IndexFlat::operator=(IndexFlat&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

IndexFlat::~IndexFlat() = default;

Expected<IndexFlat> IndexFlat::create(Context& ctx, uint32_t dimension,
                                       const IndexOptions& options) {
    IndexFlat index;
    index.impl_ = std::make_unique<Impl>();
    index.impl_->ctx = &ctx;
    index.impl_->dimension = dimension;
    index.impl_->metric = options.metric;
    index.impl_->use_fp16 = options.use_fp16;
    
    auto pipeline_result = index.init_pipelines();
    if (!pipeline_result) {
        return make_unexpected(pipeline_result.error().code(), 
                               pipeline_result.error().message());
    }
    
    return index;
}

Expected<void> IndexFlat::init_pipelines() {
    // L2 distance pipeline
    PipelineDesc l2_desc;
    l2_desc.shader_name = "distance_l2";
    l2_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},
        {1, DescriptorBinding::StorageBuffer},
        {2, DescriptorBinding::StorageBuffer}
    };
    l2_desc.push_constant_size = 16;
    
    auto l2_result = Pipeline::create(*impl_->ctx, l2_desc);
    if (!l2_result) {
        return make_unexpected(l2_result.error().code(),
                               "Failed to create L2 distance pipeline: " + l2_result.error().message());
    }
    impl_->distance_l2_pipeline = std::move(*l2_result);
    
    // Inner product pipeline
    PipelineDesc ip_desc;
    ip_desc.shader_name = "distance_ip";
    ip_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},
        {1, DescriptorBinding::StorageBuffer},
        {2, DescriptorBinding::StorageBuffer}
    };
    ip_desc.push_constant_size = 16;
    
    auto ip_result = Pipeline::create(*impl_->ctx, ip_desc);
    if (!ip_result) {
        return make_unexpected(ip_result.error().code(),
                               "Failed to create IP distance pipeline: " + ip_result.error().message());
    }
    impl_->distance_ip_pipeline = std::move(*ip_result);
    
    // Create descriptor set (same layout for both distance pipelines)
    auto desc_result = DescriptorSet::create(*impl_->ctx, impl_->distance_l2_pipeline);
    if (!desc_result) {
        return make_unexpected(desc_result.error().code(),
                               "Failed to create descriptor set");
    }
    impl_->distance_desc_set = std::move(*desc_result);
    
    // Top-k pipeline – 3 bindings: input distances, output distances, output
    // indices.  The shader computes original indices inline from workgroup/
    // thread IDs, eliminating the need for a separate sequential-index buffer.
    PipelineDesc topk_desc;
    topk_desc.shader_name = "topk_heap";
    topk_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // in_distances  (n_vectors floats)
        {1, DescriptorBinding::StorageBuffer},  // out_distances (n_wg * k floats)
        {2, DescriptorBinding::StorageBuffer},  // out_indices   (n_wg * k uints)
    };
    topk_desc.push_constant_size = 16;
    
    auto topk_result = Pipeline::create(*impl_->ctx, topk_desc);
    if (!topk_result) {
        return make_unexpected(topk_result.error().code(),
                               "Failed to create topk pipeline: " + topk_result.error().message());
    }
    impl_->topk_pipeline = std::move(*topk_result);
    
    auto topk_desc_result = DescriptorSet::create(*impl_->ctx, impl_->topk_pipeline);
    if (!topk_desc_result) {
        return make_unexpected(topk_desc_result.error().code(),
                               "Failed to create topk descriptor set");
    }
    impl_->topk_desc_set = std::move(*topk_desc_result);
    
    return {};
}

uint32_t IndexFlat::dimension() const {
    return impl_ ? impl_->dimension : 0;
}

uint64_t IndexFlat::size() const {
    return impl_ ? impl_->n_vectors : 0;
}

IndexStats IndexFlat::stats() const {
    IndexStats s{};
    if (impl_) {
        s.n_vectors = impl_->n_vectors;
        s.dimension = impl_->dimension;
        s.is_trained = true;
        s.gpu_memory_used = impl_->data_buffer.size() + impl_->ids_buffer.size();
    }
    return s;
}

Expected<void> IndexFlat::add(std::span<const float> data, uint64_t n,
                               std::span<const VectorId> ids) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }
    
    uint64_t expected_size = n * impl_->dimension;
    if (data.size() < expected_size) {
        return make_unexpected(ErrorCode::InvalidParameter, 
                               "Data size mismatch");
    }
    
    uint64_t new_size = impl_->n_vectors + n;
    
    if (new_size > impl_->capacity) {
        uint64_t new_capacity = std::max(new_size, impl_->capacity * 2);
        new_capacity = std::max(new_capacity, uint64_t(1024));
        
        auto realloc_result = reallocate_buffers(new_capacity);
        if (!realloc_result) {
            return realloc_result;
        }
        impl_->capacity = new_capacity;
    }
    
    if (impl_->use_fp16) {
        std::vector<uint8_t> fp16_data(n * impl_->dimension * sizeof(uint16_t));
        uint16_t* fp16_ptr = reinterpret_cast<uint16_t*>(fp16_data.data());
        for (uint64_t i = 0; i < n * impl_->dimension; ++i) {
            fp16_ptr[i] = float_to_half(data[i]);
        }
        auto upload_result = impl_->data_buffer.upload(fp16_data, 
            impl_->n_vectors * impl_->dimension * sizeof(uint16_t));
        if (!upload_result) {
            return upload_result;
        }
    } else {
        std::vector<uint8_t> byte_data(n * impl_->dimension * sizeof(float));
        std::memcpy(byte_data.data(), data.data(), byte_data.size());
        auto upload_result = impl_->data_buffer.upload(byte_data,
            impl_->n_vectors * impl_->dimension * sizeof(float));
        if (!upload_result) {
            return upload_result;
        }
    }
    
    if (ids.empty()) {
        for (uint64_t i = 0; i < n; ++i) {
            impl_->id_mapping.push_back(impl_->n_vectors + i);
        }
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            impl_->id_mapping.push_back(ids[i]);
        }
    }
    
    impl_->n_vectors = new_size;
    return {};
}

Expected<void> IndexFlat::reallocate_buffers(uint64_t new_capacity) {
    uint64_t element_size = impl_->use_fp16 ? sizeof(uint16_t) : sizeof(float);
    uint64_t data_size = new_capacity * impl_->dimension * element_size;
    
    BufferDesc data_desc = {};
    data_desc.size = data_size;
    data_desc.usage = BufferUsage::Storage | BufferUsage::TransferDst | BufferUsage::TransferSrc;
    data_desc.memory_type = MemoryType::DeviceLocal;
    
    auto new_data = Buffer::create(*impl_->ctx, data_desc);
    if (!new_data) {
        return make_unexpected(new_data.error().code(), 
                               "Failed to allocate data buffer");
    }
    
    if (impl_->data_buffer.valid() && impl_->n_vectors > 0) {
        auto cmd_result = CommandBuffer::create(*impl_->ctx);
        if (!cmd_result) {
            return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
        }
        auto cmd = std::move(*cmd_result);
        
        cmd.begin();
        cmd.copy_buffer(impl_->data_buffer, *new_data, 
                       impl_->n_vectors * impl_->dimension * element_size);
        cmd.end();
        
        auto submit_result = submit_and_wait(*impl_->ctx, cmd);
        if (!submit_result) {
            return submit_result;
        }
    }
    
    impl_->data_buffer = std::move(*new_data);
    return {};
}

Expected<SearchResults> IndexFlat::search(Vector query, uint32_t k) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }
    
    if (query.size() != impl_->dimension) {
        return make_unexpected(ErrorCode::InvalidDimension,
                               "Query dimension mismatch");
    }
    
    if (impl_->n_vectors == 0) {
        return SearchResults(1, k);
    }
    
    SearchResults results(1, k);
    
    auto bind_result = impl_->distance_desc_set.bind_buffer(0, impl_->data_buffer);
    if (!bind_result) {
        return make_unexpected(bind_result.error().code(), bind_result.error().message());
    }
    
    uint64_t element_size = impl_->use_fp16 ? sizeof(uint16_t) : sizeof(float);
    
    if (!impl_->query_buffer.valid() || impl_->query_buffer.size() < impl_->dimension * element_size) {
        BufferDesc query_desc = {};
        query_desc.size = impl_->dimension * element_size;
        query_desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        query_desc.memory_type = MemoryType::DeviceLocal;
        
        auto qbuf = Buffer::create(*impl_->ctx, query_desc);
        if (!qbuf) {
            return make_unexpected(qbuf.error().code(), "Failed to create query buffer");
        }
        impl_->query_buffer = std::move(*qbuf);
    }
    
    uint64_t dist_buffer_size = std::max(impl_->n_vectors * sizeof(float), uint64_t(4096));
    if (!impl_->distance_buffer.valid() || impl_->distance_buffer.size() < dist_buffer_size) {
        BufferDesc dist_desc = {};
        dist_desc.size = dist_buffer_size;
        dist_desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;
        dist_desc.memory_type = MemoryType::HostVisible;
        dist_desc.map_on_create = true;
        
        auto dbuf = Buffer::create(*impl_->ctx, dist_desc);
        if (!dbuf) {
            return make_unexpected(dbuf.error().code(), "Failed to create distance buffer");
        }
        impl_->distance_buffer = std::move(*dbuf);
    }
    
    // Upload query
    std::vector<uint8_t> query_bytes;
    if (impl_->use_fp16) {
        query_bytes.resize(impl_->dimension * sizeof(uint16_t));
        uint16_t* fp16_query = reinterpret_cast<uint16_t*>(query_bytes.data());
        for (uint32_t i = 0; i < impl_->dimension; ++i) {
            fp16_query[i] = float_to_half(query[i]);
        }
    } else {
        query_bytes.resize(impl_->dimension * sizeof(float));
        std::memcpy(query_bytes.data(), query.data(), query_bytes.size());
    }
    
    auto upload_result = impl_->query_buffer.upload(query_bytes);
    if (!upload_result) {
        return make_unexpected(upload_result.error().code(), "Failed to upload query");
    }
    
    bind_result = impl_->distance_desc_set.bind_buffer(1, impl_->query_buffer);
    if (!bind_result) {
        return make_unexpected(bind_result.error().code(), bind_result.error().message());
    }
    
    bind_result = impl_->distance_desc_set.bind_buffer(2, impl_->distance_buffer);
    if (!bind_result) {
        return make_unexpected(bind_result.error().code(), bind_result.error().message());
    }
    
    // Clear distance buffer to zeros before dispatch.
    std::vector<uint8_t> zeros(impl_->n_vectors * sizeof(float), 0);
    auto clear_result = impl_->distance_buffer.upload(zeros);
    if (!clear_result) {
        return make_unexpected(clear_result.error().code(), "Failed to clear distance buffer");
    }

    // --- Allocate / grow partial-result buffers ----------------------------
    // Pass 0 of the top-k shader writes k candidates per workgroup.
    // The host then CPU-merges those n_workgroups*k candidates.
    uint32_t n_topk_wg = (static_cast<uint32_t>(impl_->n_vectors) + TOPK_CHUNK - 1u) / TOPK_CHUNK;
    uint64_t partial_size = static_cast<uint64_t>(n_topk_wg) * k * sizeof(float);

    if (!impl_->result_distances_buffer.valid() ||
        impl_->result_distances_buffer.size() < partial_size) {

        BufferDesc res_desc = {};
        res_desc.size = partial_size;
        res_desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;
        res_desc.memory_type = MemoryType::HostVisible;
        res_desc.map_on_create = true;

        auto rdbuf = Buffer::create(*impl_->ctx, res_desc);
        if (!rdbuf) {
            return make_unexpected(rdbuf.error().code(), "Failed to create result distances buffer");
        }
        impl_->result_distances_buffer = std::move(*rdbuf);

        auto ribuf = Buffer::create(*impl_->ctx, res_desc);
        if (!ribuf) {
            return make_unexpected(ribuf.error().code(), "Failed to create result indices buffer");
        }
        impl_->result_indices_buffer = std::move(*ribuf);
    }

    // --- Bind top-k descriptor set ----------------------------------------
    auto bind_topk = impl_->topk_desc_set.bind_buffer(0, impl_->distance_buffer);
    if (!bind_topk) {
        return make_unexpected(bind_topk.error().code(), bind_topk.error().message());
    }
    bind_topk = impl_->topk_desc_set.bind_buffer(1, impl_->result_distances_buffer);
    if (!bind_topk) {
        return make_unexpected(bind_topk.error().code(), bind_topk.error().message());
    }
    bind_topk = impl_->topk_desc_set.bind_buffer(2, impl_->result_indices_buffer);
    if (!bind_topk) {
        return make_unexpected(bind_topk.error().code(), bind_topk.error().message());
    }

    // --- Record and submit the command buffer ------------------------------
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) {
        return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
    }
    auto cmd = std::move(*cmd_result);
    cmd.begin();

    // Select distance pipeline based on metric.
    Pipeline& distance_pipeline = (impl_->metric == Metric::IP)
        ? impl_->distance_ip_pipeline
        : impl_->distance_l2_pipeline;

    // Pass A: compute per-vector distances.
    cmd.bind_pipeline(distance_pipeline);
    cmd.bind_descriptor_set(distance_pipeline, impl_->distance_desc_set);

    struct DistancePushConstants {
        uint32_t n_vectors;
        uint32_t dimension;
        uint32_t query_offset;
        uint32_t pad;
    } dpc = {
        static_cast<uint32_t>(impl_->n_vectors),
        impl_->dimension,
        0u,
        0u
    };
    cmd.push_constants(distance_pipeline, &dpc, sizeof(dpc));

    uint32_t dist_wg = (static_cast<uint32_t>(impl_->n_vectors) + 255u) / 256u;
    cmd.dispatch(dist_wg);

    // Barrier: distance writes must complete before top-k reads them.
    cmd.barrier();

    // Pass B: per-workgroup bitonic-sort top-k.
    cmd.bind_pipeline(impl_->topk_pipeline);
    cmd.bind_descriptor_set(impl_->topk_pipeline, impl_->topk_desc_set);

    struct TopKPushConstants {
        uint32_t n_vectors;
        uint32_t k;
        uint32_t pad0;
        uint32_t pad1;
    } tpc = {
        static_cast<uint32_t>(impl_->n_vectors),
        k,
        0u,
        0u
    };
    cmd.push_constants(impl_->topk_pipeline, &tpc, sizeof(tpc));
    cmd.dispatch(n_topk_wg);

    cmd.barrier();
    cmd.end();

    auto submit_result = submit_and_wait(*impl_->ctx, cmd);
    if (!submit_result) {
        return make_unexpected(submit_result.error().code(), submit_result.error().message());
    }

    // --- CPU merge: find global top-k from n_topk_wg * k candidates -------
    uint32_t n_partial = n_topk_wg * k;
    std::vector<uint8_t> dist_bytes(static_cast<uint64_t>(n_partial) * sizeof(float));
    std::vector<uint8_t> idx_bytes(static_cast<uint64_t>(n_partial) * sizeof(uint32_t));

    auto download_dist = impl_->result_distances_buffer.download(dist_bytes);
    if (!download_dist) {
        return make_unexpected(download_dist.error().code(), download_dist.error().message());
    }
    auto download_idx = impl_->result_indices_buffer.download(idx_bytes);
    if (!download_idx) {
        return make_unexpected(download_idx.error().code(), download_idx.error().message());
    }

    const float*    raw_dists = reinterpret_cast<const float*>(dist_bytes.data());
    const uint32_t* raw_idxs  = reinterpret_cast<const uint32_t*>(idx_bytes.data());

    // Collect valid (distance, vector_index) pairs, discarding +inf sentinels.
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(n_partial);
    for (uint32_t i = 0; i < n_partial; ++i) {
        if (raw_idxs[i] != 0xFFFFFFFFu && raw_idxs[i] < static_cast<uint32_t>(impl_->n_vectors)) {
            candidates.emplace_back(raw_dists[i], raw_idxs[i]);
        }
    }

    uint32_t actual_k = std::min(k, static_cast<uint32_t>(candidates.size()));
    if (actual_k > 0) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + actual_k,
            candidates.end(),
            [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b) {
                return a.first < b.first;
            });

        for (uint32_t i = 0; i < actual_k; ++i) {
            results.results[i].distance = candidates[i].first;
            results.results[i].id       = impl_->id_mapping[candidates[i].second];
        }
    }

    return results;
}

Expected<SearchResults> IndexFlat::search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }
    
    SearchResults results(n_queries, k);
    
    for (uint64_t q = 0; q < n_queries; ++q) {
        Vector query(queries.data() + q * impl_->dimension, impl_->dimension);
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

Expected<void> IndexFlat::save(const std::filesystem::path& path) const {
    (void)path;
    return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

Expected<void> IndexFlat::load(const std::filesystem::path& path) {
    (void)path;
    return make_unexpected(ErrorCode::OperationFailed, "Not implemented");
}

void IndexFlat::reset() {
    if (impl_) {
        impl_->n_vectors = 0;
        impl_->id_mapping.clear();
    }
}

Expected<std::vector<float>> IndexFlat::get_vector(VectorId id) const {
    if (!impl_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }
    
    auto it = std::find(impl_->id_mapping.begin(), impl_->id_mapping.end(), id);
    if (it == impl_->id_mapping.end()) {
        return make_unexpected(ErrorCode::InvalidParameter, "ID not found");
    }
    
    std::vector<float> result(impl_->dimension);
    // TODO: Download from GPU
    return result;
}

} // namespace cw
