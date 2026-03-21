#ifndef CATWHISPER_NPU_BACKEND_HPP
#define CATWHISPER_NPU_BACKEND_HPP

#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cw {

struct NpuDeviceInfo {
    std::string name;
    std::string driver_version;
    uint64_t npu_memory = 0;        // 0 = unified memory (APU)
    bool unified_memory = false;     // true on Strix Point APU (shared DRAM)
    uint32_t num_columns = 0;        // AIE compute columns
    uint32_t tops_int8 = 0;          // peak INT8 throughput (TOPS)
};

class NpuBuffer {
public:
    NpuBuffer() = default;
    NpuBuffer(NpuBuffer&&) noexcept;
    NpuBuffer& operator=(NpuBuffer&&) noexcept;
    ~NpuBuffer();

    NpuBuffer(const NpuBuffer&) = delete;
    NpuBuffer& operator=(const NpuBuffer&) = delete;

    size_t size_bytes() const { return size_bytes_; }
    bool valid() const { return data_ != nullptr; }

    // CPU-accessible pointer (zero-copy on unified memory APUs).
    void* cpu_ptr() { return data_; }
    const void* cpu_ptr() const { return data_; }

    // Write from CPU memory into this buffer.
    void write(std::span<const std::byte> src, size_t dst_offset = 0);

    // Read from this buffer into CPU memory.
    void read(std::span<std::byte> dst, size_t src_offset = 0) const;

private:
    friend class NpuContext;
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    int dmabuf_fd_ = -1;             // DMA-BUF file descriptor (XDNA)
    bool owns_allocation_ = false;
};

class NpuContext {
public:
    NpuContext(const NpuContext&) = delete;
    NpuContext& operator=(const NpuContext&) = delete;
    NpuContext(NpuContext&&) noexcept;
    NpuContext& operator=(NpuContext&&) noexcept;
    ~NpuContext();

    // Probe whether XDNA NPU hardware + driver is available.
    [[nodiscard]] static bool available();

    // Create an NPU context. Returns error if no NPU or driver missing.
    [[nodiscard]] static Expected<NpuContext> create();

    // Device info for the detected NPU.
    const NpuDeviceInfo& device_info() const;

    // Allocate a buffer accessible by both CPU and NPU.
    // On unified-memory APUs, this is a zero-copy DMA-BUF mapping.
    [[nodiscard]] Expected<NpuBuffer> allocate(size_t bytes);

    // --- Batch compute kernels ---

    // Batch cosine distance: queries [N x dim] int8 vs database [M x dim] int8.
    // Writes distances [N x M] float into output buffer.
    // query_scale and db_scale are the inverse quantization scales
    // (i.e., the int8_scale used during quantization, typically 127.0f).
    [[nodiscard]] Expected<void> batch_cosine_distance_int8(
        const NpuBuffer& queries,     // [N x dim] int8
        const NpuBuffer& database,    // [M x dim] int8
        NpuBuffer& distances,         // [N x M] float output
        uint32_t n_queries,
        uint32_t n_database,
        uint32_t dim,
        float query_scale,
        float db_scale);

    // Batch top-k selection from distance matrix [N x M] → [N x k] results.
    [[nodiscard]] Expected<void> batch_topk(
        const NpuBuffer& distances,   // [N x M] float
        NpuBuffer& out_indices,        // [N x k] uint32
        NpuBuffer& out_values,         // [N x k] float
        uint32_t n_queries,
        uint32_t n_database,
        uint32_t k);

    // Combined: distance + top-k in single dispatch (avoids materializing full N*M matrix
    // when M is large). Uses tiled evaluation internally.
    [[nodiscard]] Expected<SearchResults> batch_search_int8(
        const NpuBuffer& queries,     // [N x dim] int8
        const NpuBuffer& database,    // [M x dim] int8
        uint32_t n_queries,
        uint32_t n_database,
        uint32_t dim,
        uint32_t k,
        float query_scale,
        float db_scale,
        Metric metric = Metric::Cosine);

    // Synchronize: wait for all pending NPU operations to complete.
    void synchronize();

    bool valid() const { return impl_ != nullptr; }

private:
    NpuContext() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw

#endif // CATWHISPER_NPU_BACKEND_HPP
