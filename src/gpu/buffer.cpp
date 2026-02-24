#define VMA_IMPLEMENTATION
#include "core/context_impl.hpp"
#include <catwhisper/buffer.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/pipeline.hpp>

namespace cw {

struct Buffer::Impl {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_info = {};
    Context* ctx = nullptr;
};

Buffer::Buffer() = default;

Buffer::Buffer(Buffer&& other) noexcept
    : impl_(std::move(other.impl_)), mapped_(other.mapped_), size_(other.size_)
{
    other.mapped_ = nullptr;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        mapped_ = other.mapped_;
        size_ = other.size_;
        other.mapped_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Buffer::~Buffer() {
    if (impl_ && impl_->ctx) {
        auto ctx_impl = impl_->ctx->impl_.get();
        if (mapped_) {
            vmaUnmapMemory(ctx_impl->allocator, impl_->allocation);
        }
        if (impl_->buffer) {
            vmaDestroyBuffer(ctx_impl->allocator, impl_->buffer, impl_->allocation);
        }
    }
}

Expected<Buffer> Buffer::create(Context& ctx, const BufferDesc& desc) {
    Buffer buffer;
    buffer.impl_ = std::make_unique<Impl>();
    buffer.impl_->ctx = &ctx;
    buffer.size_ = desc.size;
    
    VkBufferUsageFlags vk_usage = 0;
    if (has_flag(desc.usage, BufferUsage::Storage)) {
        vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (has_flag(desc.usage, BufferUsage::Uniform)) {
        vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    if (has_flag(desc.usage, BufferUsage::TransferSrc)) {
        vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (has_flag(desc.usage, BufferUsage::TransferDst)) {
        vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (has_flag(desc.usage, BufferUsage::Index)) {
        vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    if (has_flag(desc.usage, BufferUsage::Vertex)) {
        vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    
    VmaMemoryUsage vma_usage;
    VmaAllocationCreateFlags vma_flags = 0;
    
    switch (desc.memory_type) {
        case MemoryType::DeviceLocal:
            vma_usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
        case MemoryType::HostVisible:
            vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            vma_flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryType::HostCoherent:
            vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            vma_flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryType::DeviceOnly:
            vma_usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
        case MemoryType::HostReadback:
            // GPU writes, CPU reads: host-cached memory (system RAM) for fast CPU access.
            // No DEVICE_LOCAL flag so GPU writes go through PCIe; CPU reads are cached.
            vma_usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
            break;
        default:
            vma_usage = VMA_MEMORY_USAGE_GPU_ONLY;
    }
    
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = vk_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    VmaAllocationCreateInfo alloc_info = {
        .flags = vma_flags,
        .usage = vma_usage
    };
    
    VkResult result = vmaCreateBuffer(
        ctx.impl_->allocator,
        &buffer_info,
        &alloc_info,
        &buffer.impl_->buffer,
        &buffer.impl_->allocation,
        &buffer.impl_->alloc_info
    );
    
    if (result != VK_SUCCESS) {
        return make_unexpected(ErrorCode::BufferCreationFailed,
                               "Failed to create buffer");
    }
    
    if (desc.map_on_create || desc.memory_type == MemoryType::HostVisible || 
        desc.memory_type == MemoryType::HostCoherent) {
        result = vmaMapMemory(ctx.impl_->allocator, buffer.impl_->allocation, &buffer.mapped_);
        if (result != VK_SUCCESS) {
            return make_unexpected(ErrorCode::BufferCreationFailed,
                                   "Failed to map buffer memory");
        }
    }
    
    return buffer;
}

void* Buffer::vulkan_buffer() const {
    return impl_ ? reinterpret_cast<void*>(impl_->buffer) : nullptr;
}

Expected<void> Buffer::upload(std::span<const uint8_t> data, uint64_t offset) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Buffer not initialized");
    }
    
    if (offset + data.size() > size_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Upload exceeds buffer size");
    }
    
    if (mapped_) {
        std::memcpy(static_cast<uint8_t*>(mapped_) + offset, data.data(), data.size());
        return {};
    }
    
    // Need staging buffer
    BufferDesc staging_desc = {
        .size = data.size(),
        .usage = BufferUsage::TransferSrc,
        .memory_type = MemoryType::HostVisible,
        .map_on_create = true
    };
    
    auto staging = create(*impl_->ctx, staging_desc);
    if (!staging) {
        return make_unexpected(staging.error().code(), staging.error().message());
    }
    
    std::memcpy(staging->mapped_, data.data(), data.size());

    // GPU copy: staging (host-visible) → this buffer (device-local).
    auto cmd_result = CommandBuffer::create(*impl_->ctx);
    if (!cmd_result) {
        return make_unexpected(cmd_result.error().code(),
                               "Failed to create command buffer for staging upload: " +
                               cmd_result.error().message());
    }
    auto cmd = std::move(*cmd_result);
    cmd.begin();
    cmd.copy_buffer(*staging, *this, data.size(), /*src_off=*/0, offset);
    cmd.end();
    return submit_and_wait(*impl_->ctx, cmd);
}

Expected<void> Buffer::download(std::span<uint8_t> data, uint64_t offset) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "Buffer not initialized");
    }
    
    if (offset + data.size() > size_) {
        return make_unexpected(ErrorCode::InvalidParameter, "Download exceeds buffer size");
    }
    
    if (mapped_) {
        std::memcpy(data.data(), static_cast<uint8_t*>(mapped_) + offset, data.size());
        return {};
    }
    
    void* src = nullptr;
    VkResult result = vmaMapMemory(impl_->ctx->impl_->allocator, impl_->allocation, &src);
    if (result == VK_SUCCESS) {
        std::memcpy(data.data(), static_cast<uint8_t*>(src) + offset, data.size());
        vmaUnmapMemory(impl_->ctx->impl_->allocator, impl_->allocation);
        return {};
    }
    
    return make_unexpected(ErrorCode::OperationFailed, "Failed to download from buffer");
}

} // namespace cw
