#ifndef CATWHISPER_CONTEXT_HPP
#define CATWHISPER_CONTEXT_HPP

#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>

#include <cstring>
#include <memory>
#include <vector>

namespace cw {

class Buffer;
class Pipeline;
class CommandBuffer;
class DescriptorSet;

struct ContextOptions {
    int device_id = -1;
    uint64_t max_gpu_memory = 0;
    bool enable_validation = false;
    bool enable_debug_names = false;
    uint32_t num_queues = 1;
};

class Context {
public:
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;
    ~Context();

    [[nodiscard]] static Expected<Context> create(const ContextOptions& options = {});

    [[nodiscard]] static Expected<std::vector<DeviceInfo>> list_devices();

    const DeviceInfo& device_info() const;
    uint64_t total_gpu_memory() const;
    uint64_t available_gpu_memory() const;

    void synchronize();

    void* vulkan_device();
    void* vulkan_instance();
    void* vulkan_physical_device();
    uint32_t compute_queue_family() const;
    void* compute_queue();

    bool valid() const { return impl_ != nullptr; }

    friend class Buffer;
    friend class Pipeline;
    friend class CommandBuffer;
    friend class DescriptorSet;
    friend Expected<void> submit_and_wait(Context&, CommandBuffer&);

private:
    Context() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cw

#endif // CATWHISPER_CONTEXT_HPP
