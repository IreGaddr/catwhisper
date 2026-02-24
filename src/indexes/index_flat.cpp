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

struct IndexFlat::Impl {
    Context* ctx = nullptr;
    uint32_t dimension = 0;
    uint64_t n_vectors = 0;
    uint64_t capacity = 0;
    Metric metric = Metric::L2;
    bool use_fp16 = true;
    
    Buffer data_buffer;
    Buffer ids_buffer;
    
    Pipeline distance_pipeline;
    Pipeline topk_pipeline;
    DescriptorSet distance_desc_set;
    
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
    PipelineDesc distance_desc;
    distance_desc.shader_name = "distance_l2";
    distance_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},
        {1, DescriptorBinding::StorageBuffer},
        {2, DescriptorBinding::StorageBuffer}
    };
    distance_desc.push_constant_size = 16;
    
    auto distance_result = Pipeline::create(*impl_->ctx, distance_desc);
    if (!distance_result) {
        return make_unexpected(distance_result.error().code(),
                               "Failed to create distance pipeline: " + distance_result.error().message());
    }
    impl_->distance_pipeline = std::move(*distance_result);
    
    auto desc_result = DescriptorSet::create(*impl_->ctx, impl_->distance_pipeline);
    if (!desc_result) {
        return make_unexpected(desc_result.error().code(),
                               "Failed to create descriptor set");
    }
    impl_->distance_desc_set = std::move(*desc_result);
    
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
    
    // Clear distance buffer to zeros
    std::vector<uint8_t> zeros(impl_->n_vectors * sizeof(float), 0);
    auto clear_result = impl_->distance_buffer.upload(zeros);
    if (!clear_result) {
        return make_unexpected(clear_result.error().code(), "Failed to clear distance buffer");
    }
    
    // Create command buffer and dispatch compute
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) {
        return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
    }
    auto cmd = std::move(*cmd_result);
    cmd.barrier();
    cmd.bind_pipeline(impl_->distance_pipeline);
    cmd.bind_descriptor_set(impl_->distance_pipeline, impl_->distance_desc_set);
    
    struct PushConstants {
        uint32_t n_vectors;
        uint32_t dimension;
        uint32_t query_offset;
        uint32_t pad;
    } pc = {
        static_cast<uint32_t>(impl_->n_vectors),
        impl_->dimension,
        0,
        0
    };
    
    cmd.push_constants(impl_->distance_pipeline, &pc, sizeof(pc));
    
    uint32_t workgroups = (static_cast<uint32_t>(impl_->n_vectors) + 255) / 256;
    cmd.dispatch(workgroups);
    cmd.barrier();
    cmd.end();
    
    auto submit_result = submit_and_wait(*impl_->ctx, cmd);
    if (!submit_result) {
        return make_unexpected(submit_result.error().code(), submit_result.error().message());
    }
    
    // Download distances
    std::vector<uint8_t> dist_bytes(impl_->n_vectors * sizeof(float));
    auto download_result = impl_->distance_buffer.download(dist_bytes);
    if (!download_result) {
        return make_unexpected(download_result.error().code(), download_result.error().message());
    }
    
    float* distances = reinterpret_cast<float*>(dist_bytes.data());
    
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(impl_->n_vectors);
    for (uint64_t i = 0; i < impl_->n_vectors; ++i) {
        scored.emplace_back(distances[i], i);
    }
    
    uint32_t actual_k = std::min(k, static_cast<uint32_t>(impl_->n_vectors));
    std::partial_sort(scored.begin(), scored.begin() + actual_k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    
    for (uint32_t i = 0; i < actual_k; ++i) {
        results.results[i].distance = scored[i].first;
        results.results[i].id = impl_->id_mapping[scored[i].second];
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
