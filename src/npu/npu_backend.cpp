#include <catwhisper/npu_backend.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef CW_HAS_NPU
// XDNA driver headers (from Ryzen AI SDK / amdxdna kernel module).
// Provides: amdxdna device open, BO (buffer object) alloc, command submission.
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
// TODO: Include actual XDNA headers when SDK is installed:
// #include <amdxdna_accel.h>
// #include <xrt/xrt_device.h>
// #include <xrt/xrt_bo.h>
// #include <xrt/xrt_kernel.h>
#endif

namespace cw {

// ---------------------------------------------------------------------------
// NpuBuffer
// ---------------------------------------------------------------------------

NpuBuffer::NpuBuffer(NpuBuffer&& other) noexcept
    : data_(other.data_)
    , size_bytes_(other.size_bytes_)
    , dmabuf_fd_(other.dmabuf_fd_)
    , owns_allocation_(other.owns_allocation_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.dmabuf_fd_ = -1;
    other.owns_allocation_ = false;
}

NpuBuffer& NpuBuffer::operator=(NpuBuffer&& other) noexcept {
    if (this != &other) {
        // Release current resources.
        if (owns_allocation_ && data_) {
#ifdef CW_HAS_NPU
            ::munmap(data_, size_bytes_);
            if (dmabuf_fd_ >= 0) ::close(dmabuf_fd_);
#else
            std::free(data_);
#endif
        }
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        dmabuf_fd_ = other.dmabuf_fd_;
        owns_allocation_ = other.owns_allocation_;
        other.data_ = nullptr;
        other.size_bytes_ = 0;
        other.dmabuf_fd_ = -1;
        other.owns_allocation_ = false;
    }
    return *this;
}

NpuBuffer::~NpuBuffer() {
    if (owns_allocation_ && data_) {
#ifdef CW_HAS_NPU
        ::munmap(data_, size_bytes_);
        if (dmabuf_fd_ >= 0) ::close(dmabuf_fd_);
#else
        std::free(data_);
#endif
    }
}

void NpuBuffer::write(std::span<const std::byte> src, size_t dst_offset) {
    if (!data_ || dst_offset + src.size() > size_bytes_) return;
    std::memcpy(static_cast<std::byte*>(data_) + dst_offset, src.data(), src.size());
}

void NpuBuffer::read(std::span<std::byte> dst, size_t src_offset) const {
    if (!data_ || src_offset + dst.size() > size_bytes_) return;
    std::memcpy(dst.data(), static_cast<const std::byte*>(data_) + src_offset, dst.size());
}

// ---------------------------------------------------------------------------
// NpuContext::Impl
// ---------------------------------------------------------------------------

struct NpuContext::Impl {
    NpuDeviceInfo info;
#ifdef CW_HAS_NPU
    int device_fd = -1;
    // TODO: XRT device/kernel handles when SDK available.
    // xrt::device xrt_device;
    // xrt::kernel distance_kernel;
#endif

    ~Impl() {
#ifdef CW_HAS_NPU
        if (device_fd >= 0) ::close(device_fd);
#endif
    }
};

// ---------------------------------------------------------------------------
// NpuContext
// ---------------------------------------------------------------------------

NpuContext::NpuContext(NpuContext&&) noexcept = default;
NpuContext& NpuContext::operator=(NpuContext&&) noexcept = default;
NpuContext::~NpuContext() = default;

bool NpuContext::available() {
    // Probe for XDNA accelerator device nodes.
    // The amdxdna kernel module creates /dev/accel/accel0, accel1, etc.
    namespace fs = std::filesystem;
    if (!fs::exists("/dev/accel")) return false;
    for (auto& entry : fs::directory_iterator("/dev/accel")) {
        if (entry.path().filename().string().starts_with("accel")) {
            return true;
        }
    }
    return false;
}

Expected<NpuContext> NpuContext::create() {
    if (!available()) {
        return make_unexpected(ErrorCode::NoComputeCapableDevice,
            "XDNA NPU not available: no /dev/accel/accel* device found. "
            "Ensure amdxdna kernel module is loaded (modprobe amdxdna).");
    }

    NpuContext ctx;
    ctx.impl_ = std::make_unique<Impl>();

#ifdef CW_HAS_NPU
    // Open the first available accelerator device.
    namespace fs = std::filesystem;
    for (auto& entry : fs::directory_iterator("/dev/accel")) {
        auto path = entry.path().string();
        int fd = ::open(path.c_str(), O_RDWR);
        if (fd >= 0) {
            ctx.impl_->device_fd = fd;
            break;
        }
    }
    if (ctx.impl_->device_fd < 0) {
        return make_unexpected(ErrorCode::DeviceCreationFailed,
            "Failed to open XDNA accelerator device");
    }

    // TODO: Query device capabilities via ioctl / XRT API.
    // For now, populate with known Strix Point specs.
    ctx.impl_->info.name = "AMD XDNA (Strix Point)";
    ctx.impl_->info.driver_version = "1.0";  // TODO: query actual version
    ctx.impl_->info.npu_memory = 0;          // unified memory
    ctx.impl_->info.unified_memory = true;   // APU shares DRAM
    ctx.impl_->info.num_columns = 4;         // Strix Point has 4 AIE columns
    ctx.impl_->info.tops_int8 = 50;          // ~50 TOPS int8 peak

#else
    // No XDNA SDK compiled in. This path should not be reached if available()
    // returned true, but handle gracefully.
    return make_unexpected(ErrorCode::NoComputeCapableDevice,
        "XDNA NPU detected but catwhisper was built without CW_HAS_NPU. "
        "Rebuild with -DCW_ENABLE_NPU=ON and ensure XDNA SDK is installed.");
#endif

    return ctx;
}

const NpuDeviceInfo& NpuContext::device_info() const {
    return impl_->info;
}

Expected<NpuBuffer> NpuContext::allocate(size_t bytes) {
    NpuBuffer buf;
    buf.size_bytes_ = bytes;
    buf.owns_allocation_ = true;

#ifdef CW_HAS_NPU
    // On unified memory APU: allocate via DMA-BUF (zero-copy between CPU and NPU).
    // TODO: Use XDNA BO (buffer object) allocation via XRT or ioctl.
    // For now, use aligned malloc as a placeholder that works on any system.
    // When the XDNA SDK is available, this becomes:
    //   xrt::bo bo(xrt_device, bytes, xrt::bo::flags::cacheable, kernel.group_id(0));
    //   buf.data_ = bo.map();
    //   buf.dmabuf_fd_ = bo.export_buffer();

    // Placeholder: aligned allocation (works without XDNA driver for development).
    buf.data_ = std::aligned_alloc(64, (bytes + 63) & ~63ULL);
    if (!buf.data_) {
        return make_unexpected(ErrorCode::AllocationFailed,
            "NPU buffer allocation failed for " + std::to_string(bytes) + " bytes");
    }
#else
    buf.data_ = std::aligned_alloc(64, (bytes + 63) & ~63ULL);
    if (!buf.data_) {
        return make_unexpected(ErrorCode::AllocationFailed,
            "NPU buffer allocation failed for " + std::to_string(bytes) + " bytes");
    }
#endif

    return buf;
}

Expected<void> NpuContext::batch_cosine_distance_int8(
    const NpuBuffer& queries,
    const NpuBuffer& database,
    NpuBuffer& distances,
    uint32_t n_queries,
    uint32_t n_database,
    uint32_t dim,
    float query_scale,
    float db_scale) {

#ifdef CW_HAS_NPU
    // TODO: Submit int8 matmul kernel to XDNA AIE.
    // The kernel computes: for each (i, j):
    //   dot = sum_k(queries[i*dim+k] * database[j*dim+k])
    //   norm_q = sqrt(sum_k(queries[i*dim+k]^2))
    //   norm_d = sqrt(sum_k(database[j*dim+k]^2))
    //   distances[i*n_database+j] = 1.0 - dot / (norm_q * norm_d)
    //
    // On XDNA, this maps to a tiled GEMM: Q[N,dim] * D^T[dim,M] -> S[N,M]
    // followed by row/column norm division.
    //
    // For now, fall through to CPU reference implementation below.
#endif

    // CPU reference: int8 cosine distance (used until XDNA kernel is ready).
    const auto* q = static_cast<const int8_t*>(queries.cpu_ptr());
    const auto* d = static_cast<const int8_t*>(database.cpu_ptr());
    auto* out = static_cast<float*>(distances.cpu_ptr());

    const float inv_q = 1.0f / query_scale;
    const float inv_d = 1.0f / db_scale;

    #pragma omp parallel for schedule(dynamic, 4)
    for (uint32_t i = 0; i < n_queries; ++i) {
        const int8_t* qi = q + static_cast<size_t>(i) * dim;

        // Precompute query norm (dequantized).
        float norm_q_sq = 0.0f;
        for (uint32_t k = 0; k < dim; ++k) {
            float v = static_cast<float>(qi[k]) * inv_q;
            norm_q_sq += v * v;
        }
        float inv_norm_q = 1.0f / std::sqrt(norm_q_sq + 1e-30f);

        for (uint32_t j = 0; j < n_database; ++j) {
            const int8_t* dj = d + static_cast<size_t>(j) * dim;

            int32_t dot = 0;
            float norm_d_sq = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                dot += static_cast<int32_t>(qi[k]) * static_cast<int32_t>(dj[k]);
                float dv = static_cast<float>(dj[k]) * inv_d;
                norm_d_sq += dv * dv;
            }

            float dot_f = static_cast<float>(dot) * inv_q * inv_d;
            float inv_norm_d = 1.0f / std::sqrt(norm_d_sq + 1e-30f);
            out[static_cast<size_t>(i) * n_database + j] =
                1.0f - dot_f * inv_norm_q * inv_norm_d;
        }
    }

    return {};
}

Expected<void> NpuContext::batch_topk(
    const NpuBuffer& distances,
    NpuBuffer& out_indices,
    NpuBuffer& out_values,
    uint32_t n_queries,
    uint32_t n_database,
    uint32_t k) {

    const auto* dist = static_cast<const float*>(distances.cpu_ptr());
    auto* idx = static_cast<uint32_t*>(out_indices.cpu_ptr());
    auto* val = static_cast<float*>(out_values.cpu_ptr());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n_queries; ++i) {
        const float* row = dist + static_cast<size_t>(i) * n_database;
        auto* row_idx = idx + static_cast<size_t>(i) * k;
        auto* row_val = val + static_cast<size_t>(i) * k;

        // Partial sort: find k smallest distances.
        std::vector<std::pair<float, uint32_t>> candidates(n_database);
        for (uint32_t j = 0; j < n_database; ++j) {
            candidates[j] = {row[j], j};
        }
        std::partial_sort(candidates.begin(),
                          candidates.begin() + std::min(k, n_database),
                          candidates.end());

        for (uint32_t j = 0; j < k && j < n_database; ++j) {
            row_idx[j] = candidates[j].second;
            row_val[j] = candidates[j].first;
        }
    }

    return {};
}

Expected<SearchResults> NpuContext::batch_search_int8(
    const NpuBuffer& queries,
    const NpuBuffer& database,
    uint32_t n_queries,
    uint32_t n_database,
    uint32_t dim,
    uint32_t k,
    float query_scale,
    float db_scale,
    Metric metric) {

    // For large M, avoid materializing full N*M distance matrix.
    // Tile over database in chunks, maintaining running top-k per query.
    constexpr uint32_t kTileSize = 8192;

    SearchResults results(n_queries, k);

    // Per-query running top-k heaps.
    struct Candidate {
        float distance;
        uint32_t global_idx;
        bool operator>(const Candidate& o) const { return distance > o.distance; }
    };
    std::vector<std::vector<Candidate>> heaps(n_queries);

    for (uint32_t tile_start = 0; tile_start < n_database; tile_start += kTileSize) {
        uint32_t tile_count = std::min(kTileSize, n_database - tile_start);

        // In the NPU path, this would be a buffer offset; for CPU reference,
        // we iterate over the tile range directly.
        const auto* db_base = static_cast<const int8_t*>(database.cpu_ptr());

        std::vector<float> tile_distances(static_cast<size_t>(n_queries) * tile_count);

        // Compute distances for this tile.
        const auto* q = static_cast<const int8_t*>(queries.cpu_ptr());
        const float inv_q = 1.0f / query_scale;
        const float inv_d = 1.0f / db_scale;

        #pragma omp parallel for schedule(dynamic, 4)
        for (uint32_t i = 0; i < n_queries; ++i) {
            const int8_t* qi = q + static_cast<size_t>(i) * dim;

            float norm_q_sq = 0.0f;
            for (uint32_t d_ = 0; d_ < dim; ++d_) {
                float v = static_cast<float>(qi[d_]) * inv_q;
                norm_q_sq += v * v;
            }
            float inv_norm_q = 1.0f / std::sqrt(norm_q_sq + 1e-30f);

            for (uint32_t j = 0; j < tile_count; ++j) {
                const int8_t* dj = db_base + static_cast<size_t>(tile_start + j) * dim;

                int32_t dot = 0;
                float norm_d_sq = 0.0f;
                for (uint32_t d_ = 0; d_ < dim; ++d_) {
                    dot += static_cast<int32_t>(qi[d_]) * static_cast<int32_t>(dj[d_]);
                    float dv = static_cast<float>(dj[d_]) * inv_d;
                    norm_d_sq += dv * dv;
                }

                float dot_f = static_cast<float>(dot) * inv_q * inv_d;
                float inv_norm_d = 1.0f / std::sqrt(norm_d_sq + 1e-30f);
                tile_distances[static_cast<size_t>(i) * tile_count + j] =
                    1.0f - dot_f * inv_norm_q * inv_norm_d;
            }
        }

        // Merge tile results into per-query heaps.
        for (uint32_t i = 0; i < n_queries; ++i) {
            auto& heap = heaps[i];
            for (uint32_t j = 0; j < tile_count; ++j) {
                float d = tile_distances[static_cast<size_t>(i) * tile_count + j];
                uint32_t global_j = tile_start + j;

                if (heap.size() < k) {
                    heap.push_back({d, global_j});
                    if (heap.size() == k) {
                        std::make_heap(heap.begin(), heap.end(),
                                       [](auto& a, auto& b) { return a.distance < b.distance; });
                    }
                } else if (d < heap.front().distance) {
                    std::pop_heap(heap.begin(), heap.end(),
                                  [](auto& a, auto& b) { return a.distance < b.distance; });
                    heap.back() = {d, global_j};
                    std::push_heap(heap.begin(), heap.end(),
                                   [](auto& a, auto& b) { return a.distance < b.distance; });
                }
            }
        }
    }

    // Write final results.
    for (uint32_t i = 0; i < n_queries; ++i) {
        auto& heap = heaps[i];
        std::sort(heap.begin(), heap.end(),
                  [](auto& a, auto& b) { return a.distance < b.distance; });
        auto result_span = results[i];
        for (uint32_t j = 0; j < k && j < static_cast<uint32_t>(heap.size()); ++j) {
            result_span[j].id = heap[j].global_idx;
            result_span[j].distance = heap[j].distance;
        }
    }

    return results;
}

void NpuContext::synchronize() {
#ifdef CW_HAS_NPU
    // TODO: Wait for all pending XDNA command submissions.
    // xrt::run::wait() or similar.
#endif
    // CPU reference: all ops are synchronous, nothing to wait for.
}

} // namespace cw
