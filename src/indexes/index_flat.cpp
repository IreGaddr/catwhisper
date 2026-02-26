#include <catwhisper/index_flat.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/buffer.hpp>
#include <catwhisper/pipeline.hpp>
#include "core/context_impl.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>

// F16C / AVX-512 fast float32→float16 batch conversion.
// With -march=native these intrinsics map to 1-2 instructions per 8-16 floats.
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define CW_HAVE_AVX512_F16C 1
#elif defined(__F16C__)
#  include <immintrin.h>
#  define CW_HAVE_F16C 1
#endif

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

// Batch float32 → float16 conversion using F16C/AVX-512 when available,
// falling back to the scalar float_to_half otherwise.
// Destination must point into a persistently-mapped HostCoherent buffer;
// no separate staging vector is allocated.
static void convert_f32_to_f16_into(const float* __restrict__ src,
                                     uint16_t*    __restrict__ dst,
                                     uint32_t n) {
    uint32_t i = 0;
#if defined(CW_HAVE_AVX512_F16C)
    // 16 floats → 16 fp16 per iteration
    for (; i + 16u <= n; i += 16u) {
        __m512  f32 = _mm512_loadu_ps(src + i);
        __m256i f16 = _mm512_cvtps_ph(f32, _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), f16);
    }
    // 8-wide tail
    for (; i + 8u <= n; i += 8u) {
        __m256  f32 = _mm256_loadu_ps(src + i);
        __m128i f16 = _mm256_cvtps_ph(f32, _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), f16);
    }
#elif defined(CW_HAVE_F16C)
    // 8 floats → 8 fp16 per iteration
    for (; i + 8u <= n; i += 8u) {
        __m256  f32 = _mm256_loadu_ps(src + i);
        __m128i f16 = _mm256_cvtps_ph(f32, _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), f16);
    }
#endif
    // scalar tail (or full loop when no SIMD)
    for (; i < n; ++i) {
        dst[i] = float_to_half(src[i]);
    }
}

// TOPK_CHUNK must match the CHUNK constant in topk_heap.comp and distance_topk_fused.comp.
static constexpr uint32_t TOPK_CHUNK = 2048u;

// AMD-optimized: larger chunks reduce CPU merge overhead
static constexpr uint32_t AMD_CHUNK = 8192u;

struct IndexFlat::Impl {
    Context* ctx = nullptr;
    uint32_t dimension = 0;
    uint64_t n_vectors = 0;
    uint64_t capacity = 0;
    Metric metric = Metric::L2;
    bool use_fp16 = true;

    Buffer data_buffer;
    Buffer ids_buffer;

    // Legacy two-pass pipelines (kept for fallback)
    Pipeline distance_l2_pipeline;
    Pipeline distance_ip_pipeline;
    Pipeline topk_pipeline;
    DescriptorSet distance_desc_set;
    DescriptorSet topk_desc_set;

    // Fused single-pass pipeline
    Pipeline fused_l2_pipeline;
    Pipeline fused_ip_pipeline;
    DescriptorSet fused_desc_set;

    // AMD-optimized fused pipeline (larger chunks, heap-based top-k)
    Pipeline amd_l2_pipeline;
    Pipeline amd_ip_pipeline;
    DescriptorSet amd_desc_set;
    bool amd_available = false;

    // Batch search pipeline
    Pipeline batch_pipeline;
    DescriptorSet batch_desc_set;
    Buffer batch_query_buffer;
    Buffer batch_result_distances_buffer;
    Buffer batch_result_indices_buffer;

    Buffer query_buffer;
    Buffer distance_buffer;
    Buffer result_distances_buffer;
    Buffer result_indices_buffer;

    // SoA maintenance pipelines
    Pipeline transpose_add_pipeline;   // AoS staging → SoA database on add()
    DescriptorSet transpose_add_desc_set;
    Pipeline soa_repack_pipeline;      // SoA old-cap → new-cap on reallocate
    DescriptorSet soa_repack_desc_set;
    Buffer staging_buffer;             // HostCoherent mapped; holds new fp16 AoS data

    // Persistent command buffer for search (reused across calls)
    CommandBuffer search_cmd;
    bool search_cmd_valid = false;
    uint32_t search_cmd_k = 0;
    uint64_t search_cmd_n_vectors = 0;
    uint32_t last_used_k = 0;

    // Cached VkBuffer handles for the fused descriptor set.
    // vkUpdateDescriptorSets is only called when a handle actually changes,
    // avoiding 4 driver round-trips on every query in the common steady state.
    VkBuffer fused_bound_data     = VK_NULL_HANDLE;
    VkBuffer fused_bound_query    = VK_NULL_HANDLE;
    VkBuffer fused_bound_rdist    = VK_NULL_HANDLE;
    VkBuffer fused_bound_ridx     = VK_NULL_HANDLE;

    // merge_heap: pre-allocated heap for the k-way merge step.
    // merge_dist_buf / merge_idx_buf removed: we now read directly from the
    // persistently-mapped HostReadback result buffers, eliminating two memcpy
    // calls (typically ~200 bytes each) on every query.
    std::vector<std::pair<float, uint32_t>> merge_heap;

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

    // ---- Fused L2 distance + top-k pipeline ----
    PipelineDesc fused_l2_desc;
    fused_l2_desc.shader_name = "distance_topk_fused";
    fused_l2_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // database
        {1, DescriptorBinding::StorageBuffer},  // query
        {2, DescriptorBinding::StorageBuffer},  // out_distances
        {3, DescriptorBinding::StorageBuffer},  // out_indices
    };
    fused_l2_desc.push_constant_size = 20;  // n_vectors, dimension, k, metric, capacity

    auto fused_l2_result = Pipeline::create(*impl_->ctx, fused_l2_desc);
    if (!fused_l2_result) {
        return make_unexpected(fused_l2_result.error().code(),
                               "Failed to create fused L2 pipeline: " + fused_l2_result.error().message());
    }
    impl_->fused_l2_pipeline = std::move(*fused_l2_result);

    // Fused IP pipeline uses same shader with metric=1 push constant
    impl_->fused_ip_pipeline = std::move(*Pipeline::create(*impl_->ctx, fused_l2_desc));
    if (!impl_->fused_ip_pipeline.valid()) {
        return make_unexpected(ErrorCode::OperationFailed, "Failed to create fused IP pipeline");
    }

    auto fused_desc_result = DescriptorSet::create(*impl_->ctx, impl_->fused_l2_pipeline);
    if (!fused_desc_result) {
        return make_unexpected(fused_desc_result.error().code(),
                               "Failed to create fused descriptor set");
    }
    impl_->fused_desc_set = std::move(*fused_desc_result);

    // ---- Batch search pipeline (2D dispatch for multiple queries) ----
    PipelineDesc batch_desc;
    batch_desc.shader_name = "distance_topk_batch";
    batch_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // database
        {1, DescriptorBinding::StorageBuffer},  // queries (n_queries * dimension)
        {2, DescriptorBinding::StorageBuffer},  // out_distances
        {3, DescriptorBinding::StorageBuffer},  // out_indices
    };
    batch_desc.push_constant_size = 24;  // n_vectors, dimension, k, n_queries, metric, capacity

    auto batch_result = Pipeline::create(*impl_->ctx, batch_desc);
    if (!batch_result) {
        return make_unexpected(batch_result.error().code(),
                               "Failed to create batch pipeline: " + batch_result.error().message());
    }
    impl_->batch_pipeline = std::move(*batch_result);

    auto batch_desc_result = DescriptorSet::create(*impl_->ctx, impl_->batch_pipeline);
    if (!batch_desc_result) {
        return make_unexpected(batch_desc_result.error().code(),
                               "Failed to create batch descriptor set");
    }
    impl_->batch_desc_set = std::move(*batch_desc_result);

    // ---- AMD-optimized fused pipeline (larger chunks, heap-based top-k) ----
    PipelineDesc amd_l2_desc;
    amd_l2_desc.shader_name = "distance_topk_amd";
    amd_l2_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // database
        {1, DescriptorBinding::StorageBuffer},  // query
        {2, DescriptorBinding::StorageBuffer},  // out_distances
        {3, DescriptorBinding::StorageBuffer},  // out_indices
    };
    amd_l2_desc.push_constant_size = 20;  // n_vectors, dimension, k, metric, capacity

    auto amd_l2_result = Pipeline::create(*impl_->ctx, amd_l2_desc);
    if (amd_l2_result) {
        impl_->amd_l2_pipeline = std::move(*amd_l2_result);
        
        // AMD IP pipeline uses same shader with metric=1 push constant
        impl_->amd_ip_pipeline = std::move(*Pipeline::create(*impl_->ctx, amd_l2_desc));
        
        auto amd_desc_result = DescriptorSet::create(*impl_->ctx, impl_->amd_l2_pipeline);
        if (amd_desc_result) {
            impl_->amd_desc_set = std::move(*amd_desc_result);
            impl_->amd_available = true;
            fprintf(stderr, "[DEBUG] AMD pipeline loaded successfully\n");
        }
    } else {
        fprintf(stderr, "[DEBUG] AMD pipeline failed: %s\n", amd_l2_result.error().message().c_str());
    }
    // If AMD pipeline fails, we fall back to the standard fused pipeline

    // ---- SoA maintenance pipelines ----
    // Transpose add: AoS staging -> SoA database
    PipelineDesc transpose_add_desc;
    transpose_add_desc.shader_name = "transpose_add";
    transpose_add_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // staging (AoS)
        {1, DescriptorBinding::StorageBuffer},  // database (SoA)
    };
    transpose_add_desc.push_constant_size = 16;  // n_new, dim, n_existing, capacity

    auto transpose_add_result = Pipeline::create(*impl_->ctx, transpose_add_desc);
    if (!transpose_add_result) {
        return make_unexpected(transpose_add_result.error().code(),
                               "Failed to create transpose_add pipeline: " + transpose_add_result.error().message());
    }
    impl_->transpose_add_pipeline = std::move(*transpose_add_result);

    auto transpose_add_desc_result = DescriptorSet::create(*impl_->ctx, impl_->transpose_add_pipeline);
    if (!transpose_add_desc_result) {
        return make_unexpected(transpose_add_desc_result.error().code(),
                               "Failed to create transpose_add descriptor set");
    }
    impl_->transpose_add_desc_set = std::move(*transpose_add_desc_result);

    // SoA repack: old capacity -> new capacity
    PipelineDesc soa_repack_desc;
    soa_repack_desc.shader_name = "soa_repack";
    soa_repack_desc.bindings = {
        {0, DescriptorBinding::StorageBuffer},  // old_soa
        {1, DescriptorBinding::StorageBuffer},  // new_soa
    };
    soa_repack_desc.push_constant_size = 16;  // n_vectors, dim, old_capacity, new_capacity

    auto soa_repack_result = Pipeline::create(*impl_->ctx, soa_repack_desc);
    if (!soa_repack_result) {
        return make_unexpected(soa_repack_result.error().code(),
                               "Failed to create soa_repack pipeline: " + soa_repack_result.error().message());
    }
    impl_->soa_repack_pipeline = std::move(*soa_repack_result);

    auto soa_repack_desc_result = DescriptorSet::create(*impl_->ctx, impl_->soa_repack_pipeline);
    if (!soa_repack_desc_result) {
        return make_unexpected(soa_repack_desc_result.error().code(),
                               "Failed to create soa_repack descriptor set");
    }
    impl_->soa_repack_desc_set = std::move(*soa_repack_desc_result);

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
    
    // ---- Ensure staging buffer is large enough ----
    uint64_t element_size = impl_->use_fp16 ? sizeof(uint16_t) : sizeof(float);
    uint64_t staging_size = n * impl_->dimension * element_size;
    
    if (!impl_->staging_buffer.valid() || impl_->staging_buffer.size() < staging_size) {
        BufferDesc staging_desc = {};
        staging_desc.size = staging_size;
        staging_desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        staging_desc.memory_type = MemoryType::HostCoherent;
        staging_desc.map_on_create = true;
        
        auto sbuf = Buffer::create(*impl_->ctx, staging_desc);
        if (!sbuf) {
            return make_unexpected(sbuf.error().code(), "Failed to create staging buffer");
        }
        impl_->staging_buffer = std::move(*sbuf);
    }
    
    // ---- Convert to fp16 and upload to staging (AoS layout) ----
    std::vector<uint8_t> fp16_data;
    if (impl_->use_fp16) {
        fp16_data.resize(n * impl_->dimension * sizeof(uint16_t));
        uint16_t* fp16_ptr = reinterpret_cast<uint16_t*>(fp16_data.data());
        for (uint64_t i = 0; i < n * impl_->dimension; ++i) {
            fp16_ptr[i] = float_to_half(data[i]);
        }
    } else {
        fp16_data.resize(n * impl_->dimension * sizeof(float));
        std::memcpy(fp16_data.data(), data.data(), fp16_data.size());
    }
    
    auto upload_result = impl_->staging_buffer.upload(fp16_data);
    if (!upload_result) {
        return upload_result;
    }
    
    // ---- Dispatch transpose_add shader ----
    auto bind_result = impl_->transpose_add_desc_set.bind_buffer(0, impl_->staging_buffer);
    if (!bind_result) {
        return make_unexpected(bind_result.error().code(), bind_result.error().message());
    }
    bind_result = impl_->transpose_add_desc_set.bind_buffer(1, impl_->data_buffer);
    if (!bind_result) {
        return make_unexpected(bind_result.error().code(), bind_result.error().message());
    }
    
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) {
        return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
    }
    auto cmd = std::move(*cmd_result);
    
    struct TransposePushConstants {
        uint32_t n_new;
        uint32_t dim;
        uint32_t n_existing;
        uint32_t capacity;
    } tpc = {
        static_cast<uint32_t>(n),
        impl_->dimension,
        static_cast<uint32_t>(impl_->n_vectors),
        static_cast<uint32_t>(impl_->capacity)
    };
    
    cmd.begin();
    cmd.bind_pipeline(impl_->transpose_add_pipeline);
    cmd.bind_descriptor_set(impl_->transpose_add_pipeline, impl_->transpose_add_desc_set);
    cmd.push_constants(impl_->transpose_add_pipeline, &tpc, sizeof(tpc));
    
    uint64_t total_elements = n * impl_->dimension;
    uint32_t n_wg = static_cast<uint32_t>((total_elements + 255) / 256);
    cmd.dispatch(n_wg);
    cmd.end();
    
    auto submit_result = submit_and_wait(*impl_->ctx, cmd);
    if (!submit_result) {
        return submit_result;
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
        // ---- Use soa_repack shader for SoA -> SoA copy with stride change ----
        auto bind_result = impl_->soa_repack_desc_set.bind_buffer(0, impl_->data_buffer);
        if (!bind_result) {
            return make_unexpected(bind_result.error().code(), bind_result.error().message());
        }
        bind_result = impl_->soa_repack_desc_set.bind_buffer(1, *new_data);
        if (!bind_result) {
            return make_unexpected(bind_result.error().code(), bind_result.error().message());
        }
        
        auto cmd_result = CommandBuffer::create(*impl_->ctx);
        if (!cmd_result) {
            return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
        }
        auto cmd = std::move(*cmd_result);
        
        struct RepackPushConstants {
            uint32_t n_vectors;
            uint32_t dim;
            uint32_t old_capacity;
            uint32_t new_capacity;
        } rpc = {
            static_cast<uint32_t>(impl_->n_vectors),
            impl_->dimension,
            static_cast<uint32_t>(impl_->capacity),
            static_cast<uint32_t>(new_capacity)
        };
        
        cmd.begin();
        cmd.bind_pipeline(impl_->soa_repack_pipeline);
        cmd.bind_descriptor_set(impl_->soa_repack_pipeline, impl_->soa_repack_desc_set);
        cmd.push_constants(impl_->soa_repack_pipeline, &rpc, sizeof(rpc));
        
        uint64_t total_elements = impl_->n_vectors * impl_->dimension;
        uint32_t n_wg = static_cast<uint32_t>((total_elements + 255) / 256);
        cmd.dispatch(n_wg);
        cmd.end();
        
        auto submit_result = submit_and_wait(*impl_->ctx, cmd);
        if (!submit_result) {
            return submit_result;
        }
        
        impl_->fused_bound_data = VK_NULL_HANDLE;  // force descriptor rebind
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
    
    // ---- Ensure query buffer exists and is large enough ----
    uint64_t element_size = impl_->use_fp16 ? sizeof(uint16_t) : sizeof(float);
    
    if (!impl_->query_buffer.valid() || impl_->query_buffer.size() < impl_->dimension * element_size) {
        BufferDesc query_desc = {};
        query_desc.size = impl_->dimension * element_size;
        query_desc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        // HostCoherent + permanently mapped: Buffer::upload() takes the fast
        // memcpy path and never creates a staging buffer or issues a
        // submit_and_wait, cutting one full fence cycle per query.
        query_desc.memory_type = MemoryType::HostCoherent;
        query_desc.map_on_create = true;

        auto qbuf = Buffer::create(*impl_->ctx, query_desc);
        if (!qbuf) {
            return make_unexpected(qbuf.error().code(), "Failed to create query buffer");
        }
        impl_->query_buffer = std::move(*qbuf);
        impl_->fused_bound_query = VK_NULL_HANDLE;  // force descriptor rebind
    }
    
    // ---- Write query directly into the persistently-mapped HostCoherent buffer ----
    // Eliminates one heap allocation + one memcpy vs the old vector<uint8_t> approach.
    // HostCoherent writes are immediately visible to the GPU without explicit flushing.
    {
        void* mapped_q = impl_->query_buffer.mapped();
        if (impl_->use_fp16) {
            convert_f32_to_f16_into(query.data(),
                                    reinterpret_cast<uint16_t*>(mapped_q),
                                    impl_->dimension);
        } else {
            std::memcpy(mapped_q, query.data(), impl_->dimension * sizeof(float));
        }
    }
    
    // ---- Allocate partial-result buffers ----
    // Use AMD-optimized pipeline if available (larger chunks = fewer workgroups = less CPU merge)
    const uint32_t chunk_size = impl_->amd_available ? AMD_CHUNK : TOPK_CHUNK;
    uint32_t n_topk_wg = (static_cast<uint32_t>(impl_->n_vectors) + chunk_size - 1u) / chunk_size;
    uint64_t partial_size = static_cast<uint64_t>(n_topk_wg) * k * sizeof(float);

    if (!impl_->result_distances_buffer.valid() ||
        impl_->result_distances_buffer.size() < partial_size) {

        // HostReadback (VMA_MEMORY_USAGE_GPU_TO_CPU): GPU writes → CPU reads.
        // Host-cached system RAM: CPU reads are L1/L2 cached after the first access,
        // much faster than write-combined BAR memory used by HostVisible/HostCoherent.
        // We hold persistent maps and read directly from the pointers after the
        // fence/semaphore signals, eliminating all intermediate download copies.
        BufferDesc res_desc = {};
        res_desc.size = partial_size;
        res_desc.usage = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;
        res_desc.memory_type = MemoryType::HostReadback;
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

        // Invalidate cached command buffer since result buffers changed
        impl_->search_cmd_valid = false;
    }

    // ---- Update descriptor set only when VkBuffer handles change ----
    // Use AMD descriptor set if AMD pipeline is available
    DescriptorSet& active_desc_set = impl_->amd_available ? impl_->amd_desc_set : impl_->fused_desc_set;
    
    // submit_and_wait guarantees the previous submission has completed before we
    // reach this point, so the descriptor set is not "in use" and
    // vkUpdateDescriptorSets is safe.  On steady-state queries (same index, same
    // k) all four handles are identical and no driver calls are made.
    {
        auto cur_data  = reinterpret_cast<VkBuffer>(impl_->data_buffer.vulkan_buffer());
        auto cur_query = reinterpret_cast<VkBuffer>(impl_->query_buffer.vulkan_buffer());
        auto cur_rdist = reinterpret_cast<VkBuffer>(impl_->result_distances_buffer.vulkan_buffer());
        auto cur_ridx  = reinterpret_cast<VkBuffer>(impl_->result_indices_buffer.vulkan_buffer());

        if (cur_data != impl_->fused_bound_data) {
            auto r = active_desc_set.bind_buffer(0, impl_->data_buffer);
            if (!r) return make_unexpected(r.error().code(), r.error().message());
            impl_->fused_bound_data  = cur_data;
            impl_->search_cmd_valid  = false;  // dispatch count depends on n_vectors
        }
        if (cur_query != impl_->fused_bound_query) {
            auto r = active_desc_set.bind_buffer(1, impl_->query_buffer);
            if (!r) return make_unexpected(r.error().code(), r.error().message());
            impl_->fused_bound_query = cur_query;
            // Query buffer handle is constant across queries; only its contents
            // change (via memcpy into the mapped region).  The recorded cmd
            // references the descriptor set handle, not the buffer contents, so
            // cmd validity is unaffected.
        }
        if (cur_rdist != impl_->fused_bound_rdist) {
            auto r = active_desc_set.bind_buffer(2, impl_->result_distances_buffer);
            if (!r) return make_unexpected(r.error().code(), r.error().message());
            impl_->fused_bound_rdist = cur_rdist;
            impl_->search_cmd_valid  = false;
        }
        if (cur_ridx != impl_->fused_bound_ridx) {
            auto r = active_desc_set.bind_buffer(3, impl_->result_indices_buffer);
            if (!r) return make_unexpected(r.error().code(), r.error().message());
            impl_->fused_bound_ridx  = cur_ridx;
            impl_->search_cmd_valid  = false;
        }
    }

    // ---- Select pipeline based on metric and AMD availability ----
    Pipeline& fused_pipeline = impl_->amd_available
        ? ((impl_->metric == Metric::IP) ? impl_->amd_ip_pipeline : impl_->amd_l2_pipeline)
        : ((impl_->metric == Metric::IP) ? impl_->fused_ip_pipeline : impl_->fused_l2_pipeline);

    // ---- Record or reuse command buffer ----
    // need_rerecord is true on first call, when k changes, when n_vectors changes,
    // or when a buffer handle changed above (search_cmd_valid was cleared).
    bool need_rerecord = !impl_->search_cmd_valid ||
                         impl_->search_cmd_k != k ||
                         impl_->search_cmd_n_vectors != impl_->n_vectors;

    if (need_rerecord) {
        if (!impl_->search_cmd.valid()) {
            // First-ever recording: allocate a new command buffer.
            auto cmd_result = CommandBuffer::create(*impl_->ctx);
            if (!cmd_result) {
                return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
            }
            impl_->search_cmd = std::move(*cmd_result);
        } else {
            // Subsequent re-recordings: reset to Initial state.
            // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT is set on the pool.
            impl_->search_cmd.reset();
        }

        // begin_reusable() uses flags=0 (no ONE_TIME_SUBMIT_BIT) so after
        // submission+wait the buffer returns to Executable state and can be
        // re-submitted for the next query without re-recording.
        impl_->search_cmd.begin_reusable();
        impl_->search_cmd.bind_pipeline(fused_pipeline);
        impl_->search_cmd.bind_descriptor_set(fused_pipeline, active_desc_set);

        struct FusedPushConstants {
            uint32_t n_vectors;
            uint32_t dimension;
            uint32_t k;
            uint32_t metric;  // 0 = L2, 1 = IP
            uint32_t capacity;
        } fpc = {
            static_cast<uint32_t>(impl_->n_vectors),
            impl_->dimension,
            k,
            (impl_->metric == Metric::IP) ? 1u : 0u,
            static_cast<uint32_t>(impl_->capacity)
        };
        impl_->search_cmd.push_constants(fused_pipeline, &fpc, sizeof(fpc));
        impl_->search_cmd.dispatch(n_topk_wg);
        impl_->search_cmd.end();

        impl_->search_cmd_valid       = true;
        impl_->search_cmd_k           = k;
        impl_->search_cmd_n_vectors   = impl_->n_vectors;
    }
    // Else: search_cmd is in Executable state (completed after last wait, flags=0).
    // The query buffer contents were updated via memcpy into its mapped region.
    // The descriptor set still points to the same VkBuffers; GPU reads current
    // contents at execution time.  Re-submit directly.

    // ---- Submit and wait ----
    auto submit_result = submit_and_wait(*impl_->ctx, impl_->search_cmd);
    if (!submit_result) {
        return make_unexpected(submit_result.error().code(), submit_result.error().message());
    }

    // ---- CPU merge: find global top-k from n_topk_wg * k candidates ----
    // Read directly from the persistently-mapped HostReadback buffers — no download
    // memcpy, no intermediate staging.  After vkWaitSemaphores the GPU writes are
    // fully visible to the host.
    uint32_t n_partial = n_topk_wg * k;
    const float*    mapped_dists = reinterpret_cast<const float*>(
                                       impl_->result_distances_buffer.mapped());
    const uint32_t* mapped_idxs  = reinterpret_cast<const uint32_t*>(
                                       impl_->result_indices_buffer.mapped());

    // Max-heap merge: O(n log k).  Heap is max-heap with largest (worst) dist at top.
    impl_->merge_heap.clear();
    impl_->merge_heap.reserve(k + 1);

    auto heap_push = [&](float dist, uint32_t idx) {
        if (impl_->merge_heap.size() < k) {
            impl_->merge_heap.emplace_back(dist, idx);
            std::push_heap(impl_->merge_heap.begin(), impl_->merge_heap.end());
        } else if (dist < impl_->merge_heap[0].first) {
            std::pop_heap(impl_->merge_heap.begin(), impl_->merge_heap.end());
            impl_->merge_heap.back() = {dist, idx};
            std::push_heap(impl_->merge_heap.begin(), impl_->merge_heap.end());
        }
    };

    const uint32_t n_vectors_u32 = static_cast<uint32_t>(impl_->n_vectors);
    for (uint32_t i = 0; i < n_partial; ++i) {
        uint32_t idx = mapped_idxs[i];
        if (idx != 0xFFFFFFFFu && idx < n_vectors_u32) {
            heap_push(mapped_dists[i], idx);
        }
    }

    // sort_heap yields ascending order (smallest dist first)
    std::sort_heap(impl_->merge_heap.begin(), impl_->merge_heap.end());
    uint32_t actual_k = std::min(k, static_cast<uint32_t>(impl_->merge_heap.size()));
    for (uint32_t i = 0; i < actual_k; ++i) {
        results.results[i].distance = impl_->merge_heap[i].first;
        results.results[i].id       = impl_->id_mapping[impl_->merge_heap[i].second];
    }

    return results;
}

Expected<SearchResults> IndexFlat::search(std::span<const float> queries,
                                           uint64_t n_queries, uint32_t k) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Index not initialized");
    }
    
    if (impl_->n_vectors == 0) {
        return SearchResults(n_queries, k);
    }
    
    SearchResults results(n_queries, k);
    uint64_t element_size = impl_->use_fp16 ? sizeof(uint16_t) : sizeof(float);
    
    // ---- Allocate batch query buffer ----
    uint64_t batch_query_size = n_queries * impl_->dimension * element_size;
    if (!impl_->batch_query_buffer.valid() || impl_->batch_query_buffer.size() < batch_query_size) {
        BufferDesc qdesc = {};
        qdesc.size = batch_query_size;
        qdesc.usage = BufferUsage::Storage | BufferUsage::TransferDst;
        qdesc.memory_type = MemoryType::DeviceLocal;
        auto qbuf = Buffer::create(*impl_->ctx, qdesc);
        if (!qbuf) {
            return make_unexpected(qbuf.error().code(), "Failed to create batch query buffer");
        }
        impl_->batch_query_buffer = std::move(*qbuf);
    }
    
    // ---- Upload all queries ----
    std::vector<uint8_t> query_bytes;
    if (impl_->use_fp16) {
        query_bytes.resize(n_queries * impl_->dimension * sizeof(uint16_t));
        uint16_t* fp16_ptr = reinterpret_cast<uint16_t*>(query_bytes.data());
        for (uint64_t q = 0; q < n_queries; ++q) {
            for (uint32_t d = 0; d < impl_->dimension; ++d) {
                fp16_ptr[q * impl_->dimension + d] = float_to_half(queries[q * impl_->dimension + d]);
            }
        }
    } else {
        query_bytes.resize(n_queries * impl_->dimension * sizeof(float));
        std::memcpy(query_bytes.data(), queries.data(), query_bytes.size());
    }
    auto upload_result = impl_->batch_query_buffer.upload(query_bytes);
    if (!upload_result) {
        return make_unexpected(upload_result.error().code(), "Failed to upload batch queries");
    }
    
    // ---- Allocate result buffers ----
    uint32_t n_wg_x = (static_cast<uint32_t>(impl_->n_vectors) + TOPK_CHUNK - 1u) / TOPK_CHUNK;
    uint64_t partial_size = n_queries * n_wg_x * k * sizeof(float);
    
    if (!impl_->batch_result_distances_buffer.valid() ||
        impl_->batch_result_distances_buffer.size() < partial_size) {
        BufferDesc rdesc = {};
        rdesc.size = partial_size;
        rdesc.usage = BufferUsage::Storage | BufferUsage::TransferSrc;
        rdesc.memory_type = MemoryType::HostVisible;
        rdesc.map_on_create = true;
        auto rdbuf = Buffer::create(*impl_->ctx, rdesc);
        if (!rdbuf) {
            return make_unexpected(rdbuf.error().code(), "Failed to create batch result buffer");
        }
        impl_->batch_result_distances_buffer = std::move(*rdbuf);
        
        auto ribuf = Buffer::create(*impl_->ctx, rdesc);
        if (!ribuf) {
            return make_unexpected(ribuf.error().code(), "Failed to create batch index buffer");
        }
        impl_->batch_result_indices_buffer = std::move(*ribuf);
    }
    
    // ---- Bind descriptor set ----
    auto bind_result = impl_->batch_desc_set.bind_buffer(0, impl_->data_buffer);
    if (!bind_result) return make_unexpected(bind_result.error().code(), bind_result.error().message());
    bind_result = impl_->batch_desc_set.bind_buffer(1, impl_->batch_query_buffer);
    if (!bind_result) return make_unexpected(bind_result.error().code(), bind_result.error().message());
    bind_result = impl_->batch_desc_set.bind_buffer(2, impl_->batch_result_distances_buffer);
    if (!bind_result) return make_unexpected(bind_result.error().code(), bind_result.error().message());
    bind_result = impl_->batch_desc_set.bind_buffer(3, impl_->batch_result_indices_buffer);
    if (!bind_result) return make_unexpected(bind_result.error().code(), bind_result.error().message());
    
    // ---- Record and submit command buffer ----
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) return make_unexpected(cmd_result.error().code(), cmd_result.error().message());
    auto cmd = std::move(*cmd_result);
    cmd.begin();
    cmd.bind_pipeline(impl_->batch_pipeline);
    cmd.bind_descriptor_set(impl_->batch_pipeline, impl_->batch_desc_set);
    
    struct BatchPushConstants {
        uint32_t n_vectors;
        uint32_t dimension;
        uint32_t k;
        uint32_t n_queries;
        uint32_t metric;
        uint32_t capacity;
    } bpc = {
        static_cast<uint32_t>(impl_->n_vectors),
        impl_->dimension,
        k,
        static_cast<uint32_t>(n_queries),
        (impl_->metric == Metric::IP) ? 1u : 0u,
        static_cast<uint32_t>(impl_->capacity)
    };
    cmd.push_constants(impl_->batch_pipeline, &bpc, sizeof(bpc));
    cmd.dispatch(n_wg_x, static_cast<uint32_t>(n_queries), 1);
    cmd.end();
    
    auto submit_result = submit_and_wait(*impl_->ctx, cmd);
    if (!submit_result) return make_unexpected(submit_result.error().code(), submit_result.error().message());
    
    // ---- Download and merge results for each query ----
    std::vector<uint8_t> dist_bytes(partial_size);
    std::vector<uint8_t> idx_bytes(partial_size);
    
    auto download_dist = impl_->batch_result_distances_buffer.download(dist_bytes);
    if (!download_dist) return make_unexpected(download_dist.error().code(), download_dist.error().message());
    auto download_idx = impl_->batch_result_indices_buffer.download(idx_bytes);
    if (!download_idx) return make_unexpected(download_idx.error().code(), download_idx.error().message());
    
    const float* raw_dists = reinterpret_cast<const float*>(dist_bytes.data());
    const uint32_t* raw_idxs = reinterpret_cast<const uint32_t*>(idx_bytes.data());
    
    for (uint64_t q = 0; q < n_queries; ++q) {
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(n_wg_x * k);
        
        for (uint32_t wg = 0; wg < n_wg_x; ++wg) {
            uint64_t base = (q * n_wg_x + wg) * k;
            for (uint32_t i = 0; i < k; ++i) {
                uint32_t idx = raw_idxs[base + i];
                if (idx != 0xFFFFFFFFu && idx < static_cast<uint32_t>(impl_->n_vectors)) {
                    candidates.emplace_back(raw_dists[base + i], idx);
                }
            }
        }
        
        uint32_t actual_k = std::min(k, static_cast<uint32_t>(candidates.size()));
        if (actual_k > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + actual_k, candidates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            for (uint32_t i = 0; i < actual_k; ++i) {
                results.results[q * k + i].distance = candidates[i].first;
                results.results[q * k + i].id = impl_->id_mapping[candidates[i].second];
            }
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
