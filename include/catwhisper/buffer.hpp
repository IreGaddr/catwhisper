#ifndef CATWHISPER_BUFFER_HPP
#define CATWHISPER_BUFFER_HPP

#include <catwhisper/error.hpp>
#include <catwhisper/types.hpp>

#include <cstring>
#include <memory>
#include <span>

namespace cw {

class Context;

enum class BufferUsage : uint32_t {
    None = 0,
    Storage = 1 << 0,
    Uniform = 1 << 1,
    TransferSrc = 1 << 2,
    TransferDst = 1 << 3,
    Index = 1 << 4,
    Vertex = 1 << 5
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool has_flag(BufferUsage flags, BufferUsage flag) {
    return (flags & flag) == flag;
}

enum class MemoryType {
    DeviceLocal,
    HostVisible,
    HostCoherent,
    DeviceOnly
};

struct BufferDesc {
    uint64_t size;
    BufferUsage usage;
    MemoryType memory_type;
    bool map_on_create = false;
};

class Buffer {
public:
    Buffer();
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    ~Buffer();

    static Expected<Buffer> create(Context& ctx, const BufferDesc& desc);

    void* mapped() const { return mapped_; }
    uint64_t size() const { return size_; }
    bool valid() const { return impl_ != nullptr; }

    void* vulkan_buffer() const;

    Expected<void> upload(std::span<const uint8_t> data, uint64_t offset = 0);
    Expected<void> download(std::span<uint8_t> data, uint64_t offset = 0);

    template<typename T>
    Expected<void> upload_typed(std::span<const T> data, uint64_t offset_elements = 0) {
        uint64_t byte_offset = offset_elements * sizeof(T);
        std::vector<uint8_t> bytes(data.size() * sizeof(T));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        return upload(bytes, byte_offset);
    }

    template<typename T>
    Expected<void> download_typed(std::span<T> data, uint64_t offset_elements = 0) {
        uint64_t byte_offset = offset_elements * sizeof(T);
        std::vector<uint8_t> bytes(data.size() * sizeof(T));
        auto result = download(bytes, byte_offset);
        if (result) {
            std::memcpy(data.data(), bytes.data(), bytes.size());
        }
        return result;
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void* mapped_ = nullptr;
    uint64_t size_ = 0;
};

} // namespace cw

#endif // CATWHISPER_BUFFER_HPP
