#include <catwhisper/index_flat_npu.hpp>
#include <catwhisper/distance.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <shared_mutex>
#include <vector>

namespace cw {

struct IndexFlatNpu::Impl {
    NpuContext* npu_ctx = nullptr;
    IndexFlatNpuOptions options;
    uint32_t dim = 0;

    // CPU-side storage: float vectors + int8 quantized copies.
    std::vector<float> vectors;       // [n_vectors x dim] float32
    std::vector<int8_t> vectors_int8; // [n_vectors x dim] int8
    std::vector<VectorId> ids;
    uint64_t n_vectors = 0;
    uint64_t capacity = 0;

    // NPU-side persistent database buffer (int8).
    NpuBuffer npu_db_buffer;
    uint64_t npu_synced_count = 0;    // How many vectors are synced to NPU.

    mutable std::shared_mutex mutex;

    void quantize_range(uint64_t start, uint64_t end) {
        const float scale = options.int8_scale;
        const size_t stride = dim;
        for (uint64_t i = start; i < end; ++i) {
            const float* src = vectors.data() + i * stride;
            int8_t* dst = vectors_int8.data() + i * stride;
            for (uint32_t d = 0; d < dim; ++d) {
                float v = std::round(src[d] * scale);
                v = std::clamp(v, -127.0f, 127.0f);
                dst[d] = static_cast<int8_t>(v);
            }
        }
    }

    void ensure_capacity(uint64_t needed) {
        if (needed <= capacity) return;
        uint64_t new_cap = std::max(needed, capacity * 2);
        new_cap = std::max(new_cap, uint64_t{1024});
        vectors.resize(new_cap * dim);
        vectors_int8.resize(new_cap * dim);
        ids.resize(new_cap);
        capacity = new_cap;
    }
};

Expected<IndexFlatNpu> IndexFlatNpu::create(
    NpuContext& npu_ctx,
    uint32_t dimension,
    const IndexFlatNpuOptions& options) {

    if (dimension == 0) {
        return make_unexpected(ErrorCode::InvalidDimension, "Dimension must be > 0");
    }

    IndexFlatNpu idx;
    idx.impl_ = std::make_unique<Impl>();
    idx.impl_->npu_ctx = &npu_ctx;
    idx.impl_->options = options;
    idx.impl_->dim = dimension;
    return idx;
}

IndexFlatNpu::IndexFlatNpu(IndexFlatNpu&&) noexcept = default;
IndexFlatNpu& IndexFlatNpu::operator=(IndexFlatNpu&&) noexcept = default;
IndexFlatNpu::~IndexFlatNpu() = default;

uint32_t IndexFlatNpu::dimension() const { return impl_ ? impl_->dim : 0; }
uint64_t IndexFlatNpu::size() const { return impl_ ? impl_->n_vectors : 0; }

IndexStats IndexFlatNpu::stats() const {
    if (!impl_) return {};
    IndexStats s;
    s.n_vectors = impl_->n_vectors;
    s.dimension = impl_->dim;
    s.memory_used = impl_->n_vectors * impl_->dim * (sizeof(float) + sizeof(int8_t));
    s.gpu_memory_used = impl_->npu_db_buffer.size_bytes();
    s.is_trained = true;
    return s;
}

Expected<void> IndexFlatNpu::add(std::span<const float> data, uint64_t n,
                                  std::span<const VectorId> ext_ids) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");
    if (data.size() < n * impl_->dim) {
        return make_unexpected(ErrorCode::InvalidParameter, "Data buffer too small");
    }

    std::unique_lock lock(impl_->mutex);

    uint64_t old_count = impl_->n_vectors;
    impl_->ensure_capacity(old_count + n);

    // Copy float vectors.
    std::memcpy(impl_->vectors.data() + old_count * impl_->dim,
                data.data(),
                n * impl_->dim * sizeof(float));

    // Assign IDs.
    for (uint64_t i = 0; i < n; ++i) {
        impl_->ids[old_count + i] = (ext_ids.empty() || i >= ext_ids.size())
            ? (old_count + i)
            : ext_ids[i];
    }

    impl_->n_vectors = old_count + n;

    // Quantize new vectors to int8.
    impl_->quantize_range(old_count, impl_->n_vectors);

    return {};
}

Expected<SearchResults> IndexFlatNpu::search(Vector query, uint32_t k) {
    // Single query: CPU brute-force. NPU dispatch overhead not worth it.
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);

    uint64_t n = impl_->n_vectors;
    if (n == 0) return SearchResults(1, k);

    uint32_t dim = impl_->dim;
    const float* db = impl_->vectors.data();

    // Compute distances on CPU.
    std::vector<std::pair<float, uint64_t>> candidates(n);

    #pragma omp parallel for schedule(static) if(n > 4096)
    for (uint64_t i = 0; i < n; ++i) {
        std::span<const float> a(query.data(), dim);
        std::span<const float> b(db + i * dim, dim);
        candidates[i] = {distance::cosine_distance(a, b), i};
    }

    uint32_t actual_k = std::min(k, static_cast<uint32_t>(n));
    std::partial_sort(candidates.begin(),
                      candidates.begin() + actual_k,
                      candidates.end());

    SearchResults results(1, k);
    for (uint32_t j = 0; j < actual_k; ++j) {
        results[0][j].id = impl_->ids[candidates[j].second];
        results[0][j].distance = candidates[j].first;
    }
    return results;
}

Expected<SearchResults> IndexFlatNpu::search(std::span<const float> queries,
                                              uint64_t n_queries, uint32_t k) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);

    uint64_t n_db = impl_->n_vectors;
    if (n_db == 0) return SearchResults(static_cast<uint32_t>(n_queries), k);

    uint32_t dim = impl_->dim;

    // Route to NPU if batch is large enough and NPU buffer is synced.
    if (n_queries >= impl_->options.batch_threshold &&
        impl_->npu_db_buffer.valid() &&
        impl_->npu_synced_count == n_db) {

        // Quantize query vectors to int8.
        std::vector<int8_t> q_int8(n_queries * dim);
        const float scale = impl_->options.int8_scale;
        for (uint64_t i = 0; i < n_queries * dim; ++i) {
            float v = std::round(queries[i] * scale);
            v = std::clamp(v, -127.0f, 127.0f);
            q_int8[i] = static_cast<int8_t>(v);
        }

        // Allocate query buffer on NPU.
        auto q_buf = impl_->npu_ctx->allocate(n_queries * dim);
        if (!q_buf) return std::unexpected(q_buf.error());

        q_buf->write(std::as_bytes(std::span(q_int8)));

        // Dispatch batch search.
        auto npu_results = impl_->npu_ctx->batch_search_int8(
            *q_buf,
            impl_->npu_db_buffer,
            static_cast<uint32_t>(n_queries),
            static_cast<uint32_t>(n_db),
            dim, k,
            scale, scale,
            Metric::Cosine);

        if (!npu_results) return std::unexpected(npu_results.error());

        // Remap internal indices to VectorIds.
        SearchResults results = std::move(*npu_results);
        for (uint32_t i = 0; i < n_queries; ++i) {
            for (uint32_t j = 0; j < k; ++j) {
                auto& r = results[i][j];
                if (r.id < n_db) {
                    r.id = impl_->ids[r.id];
                }
            }
        }
        return results;
    }

    // CPU fallback: brute-force with float32 vectors.
    SearchResults results(static_cast<uint32_t>(n_queries), k);

    #pragma omp parallel for schedule(dynamic, 1)
    for (uint64_t qi = 0; qi < n_queries; ++qi) {
        std::span<const float> q(queries.data() + qi * dim, dim);

        std::vector<std::pair<float, uint64_t>> candidates(n_db);
        const float* db = impl_->vectors.data();
        for (uint64_t j = 0; j < n_db; ++j) {
            std::span<const float> d(db + j * dim, dim);
            candidates[j] = {distance::cosine_distance(q, d), j};
        }

        uint32_t actual_k = std::min(k, static_cast<uint32_t>(n_db));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + actual_k,
                          candidates.end());

        for (uint32_t j = 0; j < actual_k; ++j) {
            results[static_cast<uint32_t>(qi)][j].id =
                impl_->ids[candidates[j].second];
            results[static_cast<uint32_t>(qi)][j].distance =
                candidates[j].first;
        }
    }

    return results;
}

Expected<void> IndexFlatNpu::sync() {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::unique_lock lock(impl_->mutex);

    uint64_t n = impl_->n_vectors;
    if (n == 0) return {};

    uint32_t dim = impl_->dim;
    size_t needed_bytes = n * dim * sizeof(int8_t);

    // Reallocate NPU buffer if needed.
    if (!impl_->npu_db_buffer.valid() || impl_->npu_db_buffer.size_bytes() < needed_bytes) {
        // Allocate with headroom for growth (2x).
        size_t alloc_bytes = needed_bytes * 2;
        auto buf = impl_->npu_ctx->allocate(alloc_bytes);
        if (!buf) return std::unexpected(buf.error());
        impl_->npu_db_buffer = std::move(*buf);
    }

    // Upload all int8 vectors to NPU buffer.
    impl_->npu_db_buffer.write(
        std::as_bytes(std::span(impl_->vectors_int8.data(),
                                static_cast<size_t>(n) * dim)),
        0);

    impl_->npu_synced_count = n;
    return {};
}

bool IndexFlatNpu::needs_sync() const {
    if (!impl_) return false;
    return impl_->npu_synced_count < impl_->n_vectors;
}

Expected<void> IndexFlatNpu::save(const std::filesystem::path& path) const {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::shared_lock lock(impl_->mutex);

    std::ofstream out(path, std::ios::binary);
    if (!out) return make_unexpected(ErrorCode::WriteFailed, "Cannot open " + path.string());

    // Header.
    uint32_t magic = 0x4E505546; // "NPUF"
    uint32_t version = 1;
    uint32_t dim = impl_->dim;
    uint64_t n = impl_->n_vectors;
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(&dim), 4);
    out.write(reinterpret_cast<const char*>(&n), 8);

    // IDs.
    out.write(reinterpret_cast<const char*>(impl_->ids.data()),
              static_cast<std::streamsize>(n * sizeof(VectorId)));

    // Float vectors.
    out.write(reinterpret_cast<const char*>(impl_->vectors.data()),
              static_cast<std::streamsize>(n * dim * sizeof(float)));

    return {};
}

Expected<void> IndexFlatNpu::load(const std::filesystem::path& path) {
    if (!impl_) return make_unexpected(ErrorCode::OperationFailed, "Index not initialized");

    std::unique_lock lock(impl_->mutex);

    std::ifstream in(path, std::ios::binary);
    if (!in) return make_unexpected(ErrorCode::FileNotFound, "Cannot open " + path.string());

    uint32_t magic, version, dim;
    uint64_t n;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    in.read(reinterpret_cast<char*>(&n), 8);

    if (magic != 0x4E505546 || version != 1 || dim != impl_->dim) {
        return make_unexpected(ErrorCode::InvalidFileFormat, "Bad NPU index file");
    }

    impl_->ensure_capacity(n);
    impl_->n_vectors = n;

    in.read(reinterpret_cast<char*>(impl_->ids.data()),
            static_cast<std::streamsize>(n * sizeof(VectorId)));
    in.read(reinterpret_cast<char*>(impl_->vectors.data()),
            static_cast<std::streamsize>(n * dim * sizeof(float)));

    // Requantize.
    impl_->quantize_range(0, n);
    impl_->npu_synced_count = 0; // Needs sync after load.

    return {};
}

void IndexFlatNpu::reset() {
    if (!impl_) return;
    std::unique_lock lock(impl_->mutex);
    impl_->n_vectors = 0;
    impl_->npu_synced_count = 0;
    impl_->npu_db_buffer = NpuBuffer{}; // Release NPU buffer.
}

} // namespace cw
