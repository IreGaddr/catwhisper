#include <catwhisper/pipeline.hpp>
#include <catwhisper/context.hpp>
#include <catwhisper/buffer.hpp>
#include "core/context_impl.hpp"

#include <fstream>
#include <vector>

namespace cw {

struct Pipeline::Impl {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    uint32_t push_constant_size = 0;
    Context* ctx = nullptr;
};

Pipeline::Pipeline() = default;

Pipeline::Pipeline(Pipeline&& other) noexcept
    : impl_(std::move(other.impl_)) {}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

Pipeline::~Pipeline() {
    if (impl_ && impl_->ctx) {
        VkDevice dev = impl_->ctx->impl_->device;
        if (impl_->pipeline) {
            vkDestroyPipeline(dev, impl_->pipeline, nullptr);
        }
        if (impl_->layout) {
            vkDestroyPipelineLayout(dev, impl_->layout, nullptr);
        }
        if (impl_->set_layout) {
            vkDestroyDescriptorSetLayout(dev, impl_->set_layout, nullptr);
        }
    }
}

Expected<Pipeline> Pipeline::create(Context& ctx, const PipelineDesc& desc) {
    Pipeline pipeline;
    pipeline.impl_ = std::make_unique<Impl>();
    pipeline.impl_->ctx = &ctx;
    pipeline.impl_->push_constant_size = desc.push_constant_size;
    
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto& b : desc.bindings) {
        VkDescriptorType vk_type;
        switch (b.type) {
            case DescriptorBinding::StorageBuffer:
                vk_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case DescriptorBinding::UniformBuffer:
                vk_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            default:
                vk_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        
        bindings.push_back({
            .binding = b.binding,
            .descriptorType = vk_type,
            .descriptorCount = b.count,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        });
    }
    
    VkDescriptorSetLayoutCreateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    
    if (vkCreateDescriptorSetLayout(ctx.impl_->device, &set_info, nullptr,
                                    &pipeline.impl_->set_layout) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::PipelineCreationFailed,
                               "Failed to create descriptor set layout");
    }
    
    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = desc.push_constant_size
    };
    
    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pipeline.impl_->set_layout,
        .pushConstantRangeCount = desc.push_constant_size > 0 ? 1u : 0u,
        .pPushConstantRanges = desc.push_constant_size > 0 ? &push_range : nullptr
    };
    
    if (vkCreatePipelineLayout(ctx.impl_->device, &layout_info, nullptr,
                               &pipeline.impl_->layout) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::PipelineCreationFailed,
                               "Failed to create pipeline layout");
    }
    
    std::vector<uint32_t> spirv;
    auto result = load_shader(ctx, desc.shader_name, spirv);
    if (!result) {
        return make_unexpected(result.error().code(), result.error().message());
    }
    
    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode = spirv.data()
    };
    
    VkShaderModule shader_module;
    if (vkCreateShaderModule(ctx.impl_->device, &shader_info, nullptr, &shader_module) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::ShaderCompilationFailed,
                               "Failed to create shader module");
    }
    
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main"
        },
        .layout = pipeline.impl_->layout
    };
    
    VkResult vk_result = vkCreateComputePipelines(ctx.impl_->device, VK_NULL_HANDLE, 1,
                                                   &pipeline_info, nullptr,
                                                   &pipeline.impl_->pipeline);
    
    vkDestroyShaderModule(ctx.impl_->device, shader_module, nullptr);
    
    if (vk_result != VK_SUCCESS) {
        return make_unexpected(ErrorCode::PipelineCreationFailed,
                               "Failed to create compute pipeline");
    }
    
    return pipeline;
}

Expected<void> Pipeline::load_shader(Context& ctx, const std::string& name,
                                      std::vector<uint32_t>& spirv) {
    std::string path = "shaders/" + name + ".comp.spv";
    
    // Try multiple paths
    std::vector<std::string> search_paths = {
        path,
        "build_release/shaders/" + name + ".comp.spv",
        "build/shaders/" + name + ".comp.spv",
        "../shaders/" + name + ".comp.spv",
        "../build_release/shaders/" + name + ".comp.spv",
        "../build/shaders/" + name + ".comp.spv"
    };
    
    std::ifstream file;
    for (const auto& p : search_paths) {
        file.open(p, std::ios::binary);
        if (file.is_open()) break;
    }
    
    if (!file.is_open()) {
        return make_unexpected(ErrorCode::FileNotFound,
                               "Shader file not found: " + name);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    spirv.resize(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    
    if (!file) {
        return make_unexpected(ErrorCode::ReadFailed,
                               "Failed to read shader file");
    }
    
    return {};
}

void* Pipeline::vulkan_pipeline() const {
    return impl_ ? reinterpret_cast<void*>(impl_->pipeline) : nullptr;
}

void* Pipeline::vulkan_pipeline_layout() const {
    return impl_ ? reinterpret_cast<void*>(impl_->layout) : nullptr;
}

void* Pipeline::vulkan_descriptor_set_layout() const {
    return impl_ ? reinterpret_cast<void*>(impl_->set_layout) : nullptr;
}

uint32_t Pipeline::push_constant_size() const {
    return impl_ ? impl_->push_constant_size : 0;
}

struct DescriptorSet::Impl {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Context* ctx = nullptr;
};

DescriptorSet::DescriptorSet() = default;

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : impl_(std::move(other.impl_)) {}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

DescriptorSet::~DescriptorSet() {
    if (impl_ && impl_->ctx) {
        VkDevice dev = impl_->ctx->impl_->device;
        if (impl_->pool) {
            vkDestroyDescriptorPool(dev, impl_->pool, nullptr);
        }
    }
}

Expected<DescriptorSet> DescriptorSet::create(Context& ctx, const Pipeline& pipeline) {
    DescriptorSet set;
    set.impl_ = std::make_unique<Impl>();
    set.impl_->ctx = &ctx;
    
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16
    };
    
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size
    };
    
    if (vkCreateDescriptorPool(ctx.impl_->device, &pool_info, nullptr,
                               &set.impl_->pool) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::OperationFailed,
                               "Failed to create descriptor pool");
    }
    
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = set.impl_->pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipeline.impl_->set_layout
    };
    
    if (vkAllocateDescriptorSets(ctx.impl_->device, &alloc_info, &set.impl_->set) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::OperationFailed,
                               "Failed to allocate descriptor set");
    }
    
    return set;
}

Expected<void> DescriptorSet::bind_buffer(uint32_t binding, const Buffer& buffer) {
    if (!impl_ || !impl_->ctx) {
        return make_unexpected(ErrorCode::InvalidParameter, "DescriptorSet not initialized");
    }
    
    VkDescriptorBufferInfo buffer_info = {
        .buffer = reinterpret_cast<VkBuffer>(buffer.vulkan_buffer()),
        .offset = 0,
        .range = buffer.size()
    };
    
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = impl_->set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info
    };
    
    vkUpdateDescriptorSets(impl_->ctx->impl_->device, 1, &write, 0, nullptr);
    return {};
}

void* DescriptorSet::vulkan_descriptor_set() const {
    return impl_ ? reinterpret_cast<void*>(impl_->set) : nullptr;
}

void* DescriptorSet::vulkan_descriptor_pool() const {
    return impl_ ? reinterpret_cast<void*>(impl_->pool) : nullptr;
}

struct CommandBuffer::Impl {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    Context* ctx = nullptr;
};

CommandBuffer::CommandBuffer() = default;

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : impl_(std::move(other.impl_)) {}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

CommandBuffer::~CommandBuffer() {
    if (impl_ && impl_->ctx && impl_->cmd) {
        vkFreeCommandBuffers(impl_->ctx->impl_->device,
                             impl_->ctx->impl_->command_pool, 1, &impl_->cmd);
    }
}

Expected<CommandBuffer> CommandBuffer::create(Context& ctx) {
    CommandBuffer cmd;
    cmd.impl_ = std::make_unique<Impl>();
    cmd.impl_->ctx = &ctx;
    
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.impl_->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    
    if (vkAllocateCommandBuffers(ctx.impl_->device, &alloc_info, &cmd.impl_->cmd) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::OperationFailed,
                               "Failed to allocate command buffer");
    }
    
    return cmd;
}

void CommandBuffer::begin() {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(impl_->cmd, &begin_info);
}

void CommandBuffer::begin_reusable() {
    // flags = 0: no ONE_TIME_SUBMIT_BIT.  After the fence signals the buffer
    // returns to Executable state and may be re-submitted without re-recording.
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0
    };
    vkBeginCommandBuffer(impl_->cmd, &begin_info);
}

void CommandBuffer::reset() {
    // Transitions Executable/Invalid → Initial so the buffer can be re-recorded.
    // The pool must have VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT.
    vkResetCommandBuffer(impl_->cmd, 0);
}

void CommandBuffer::end() {
    vkEndCommandBuffer(impl_->cmd);
}

void CommandBuffer::bind_pipeline(const Pipeline& pipeline) {
    vkCmdBindPipeline(impl_->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      reinterpret_cast<VkPipeline>(pipeline.vulkan_pipeline()));
}

void CommandBuffer::bind_descriptor_set(const Pipeline& pipeline, const DescriptorSet& set) {
    VkDescriptorSet vk_set = reinterpret_cast<VkDescriptorSet>(set.vulkan_descriptor_set());
    vkCmdBindDescriptorSets(impl_->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            reinterpret_cast<VkPipelineLayout>(pipeline.vulkan_pipeline_layout()),
                            0, 1, &vk_set, 0, nullptr);
}

void CommandBuffer::push_constants(const Pipeline& pipeline, const void* data, uint32_t size) {
    vkCmdPushConstants(impl_->cmd,
                       reinterpret_cast<VkPipelineLayout>(pipeline.vulkan_pipeline_layout()),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(impl_->cmd, x, y, z);
}

void CommandBuffer::copy_buffer(const Buffer& src, Buffer& dst, uint64_t size,
                                 uint64_t src_offset, uint64_t dst_offset) {
    VkBufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = size
    };
    vkCmdCopyBuffer(impl_->cmd,
                    reinterpret_cast<VkBuffer>(src.vulkan_buffer()),
                    reinterpret_cast<VkBuffer>(dst.vulkan_buffer()),
                    1, &copy);
}

void CommandBuffer::barrier() {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT
    };
    
    vkCmdPipelineBarrier(impl_->cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void* CommandBuffer::vulkan_command_buffer() const {
    return impl_ ? reinterpret_cast<void*>(impl_->cmd) : nullptr;
}

Expected<void> submit_and_wait(Context& ctx, CommandBuffer& cmd) {
    // Use the persistent timeline semaphore — no vkResetFences needed.
    uint64_t signal_value = ++ctx.impl_->compute_timeline_value;

    VkTimelineSemaphoreSubmitInfo timeline_submit = {};
    timeline_submit.sType                    = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit.signalSemaphoreValueCount = 1;
    timeline_submit.pSignalSemaphoreValues   = &signal_value;

    VkSubmitInfo submit_info = {};
    submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext                = &timeline_submit;
    submit_info.commandBufferCount   = 1;
    submit_info.pCommandBuffers      = &cmd.impl_->cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores    = &ctx.impl_->compute_timeline_semaphore;

    if (vkQueueSubmit(ctx.impl_->compute_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
        return make_unexpected(ErrorCode::OperationFailed, "Failed to submit queue");
    }

    // Userspace spin-poll on the timeline semaphore counter.
    // On NVIDIA's proprietary driver vkGetSemaphoreCounterValue reads from a
    // GPU-mapped surface in process address space — no kernel transition needed.
    // This eliminates the ~5-15 μs kernel sleep/wake-up latency that
    // vkWaitSemaphores incurs for short-running GPU workloads.
    //
    // SPIN_LIMIT caps the busy-wait so that for very long GPU workloads we fall
    // back to the blocking path and don't pin a CPU core forever.
    static constexpr uint32_t SPIN_LIMIT = 2000000u;
    uint64_t current = 0;
    for (uint32_t spin = 0; spin < SPIN_LIMIT; ++spin) {
        if (spin > 0) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();  // PAUSE hint: reduces power and improves SMT throughput
#endif
        }
        if (vkGetSemaphoreCounterValue(ctx.impl_->device,
                                       ctx.impl_->compute_timeline_semaphore,
                                       &current) == VK_SUCCESS
            && current >= signal_value) {
            return {};
        }
    }

    // Fallback: blocking wait (should only trigger for very long GPU workloads)
    VkSemaphoreWaitInfo wait_info = {};
    wait_info.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores    = &ctx.impl_->compute_timeline_semaphore;
    wait_info.pValues        = &signal_value;

    vkWaitSemaphores(ctx.impl_->device, &wait_info, UINT64_MAX);
    return {};
}

} // namespace cw
