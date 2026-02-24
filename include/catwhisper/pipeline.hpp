#ifndef CATWHISPER_PIPELINE_HPP
#define CATWHISPER_PIPELINE_HPP

#include <catwhisper/error.hpp>

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace cw {

class Context;
class Buffer;
class CommandBuffer;

struct DescriptorBinding {
    uint32_t binding;
    enum Type { StorageBuffer, UniformBuffer, CombinedImageSampler } type;
    uint32_t count = 1;
};

struct PipelineDesc {
    std::string shader_name;
    std::vector<DescriptorBinding> bindings;
    uint32_t push_constant_size = 0;
};

class Pipeline {
public:
    Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    ~Pipeline();

    static Expected<Pipeline> create(Context& ctx, const PipelineDesc& desc);

    void* vulkan_pipeline() const;
    void* vulkan_pipeline_layout() const;
    void* vulkan_descriptor_set_layout() const;
    uint32_t push_constant_size() const;

    bool valid() const { return impl_ != nullptr; }

    friend class DescriptorSet;
    friend Expected<void> submit_and_wait(Context&, CommandBuffer&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    static Expected<void> load_shader(Context& ctx, const std::string& name,
                                      std::vector<uint32_t>& spirv);
};

struct DescriptorSet {
public:
    DescriptorSet();
    DescriptorSet(DescriptorSet&&) noexcept;
    DescriptorSet& operator=(DescriptorSet&&) noexcept;
    ~DescriptorSet();

    static Expected<DescriptorSet> create(Context& ctx, const Pipeline& pipeline);

    Expected<void> bind_buffer(uint32_t binding, const Buffer& buffer);

    void* vulkan_descriptor_set() const;
    void* vulkan_descriptor_pool() const;

    bool valid() const { return impl_ != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class CommandBuffer {
public:
    CommandBuffer();
    CommandBuffer(CommandBuffer&&) noexcept;
    CommandBuffer& operator=(CommandBuffer&&) noexcept;
    ~CommandBuffer();

    static Expected<CommandBuffer> create(Context& ctx);

    void begin();
    void end();

    void bind_pipeline(const Pipeline& pipeline);
    void bind_descriptor_set(const Pipeline& pipeline, const DescriptorSet& set);
    void push_constants(const Pipeline& pipeline, const void* data, uint32_t size);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);

    void copy_buffer(const Buffer& src, Buffer& dst, uint64_t size,
                     uint64_t src_offset = 0, uint64_t dst_offset = 0);

    void barrier();

    void* vulkan_command_buffer() const;

    bool valid() const { return impl_ != nullptr; }

    friend Expected<void> submit_and_wait(Context&, CommandBuffer&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Expected<void> submit_and_wait(Context& ctx, CommandBuffer& cmd);

} // namespace cw

#endif // CATWHISPER_PIPELINE_HPP
