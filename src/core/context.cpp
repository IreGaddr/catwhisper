#include "context_impl.hpp"

#include <catwhisper/context.hpp>

#include <algorithm>
#include <cstring>

namespace cw {

Context::Context(Context&& other) noexcept
    : impl_(std::move(other.impl_)) {}

Context& Context::operator=(Context&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

Context::~Context() = default;

Expected<Context> Context::create(const ContextOptions& options) {
    Context ctx;
    ctx.impl_ = std::make_unique<Impl>();
    
    auto result = ctx.impl_->init(options);
    if (!result) {
        return make_unexpected(result.error().code(), result.error().message());
    }
    
    return ctx;
}

Expected<std::vector<DeviceInfo>> Context::list_devices() {
    return Impl::list_devices();
}

const DeviceInfo& Context::device_info() const {
    return impl_->device_info;
}

uint64_t Context::total_gpu_memory() const {
    return impl_->device_info.total_memory;
}

uint64_t Context::available_gpu_memory() const {
    return impl_->available_memory();
}

void Context::synchronize() {
    impl_->synchronize();
}

void* Context::vulkan_device() {
    return impl_ ? reinterpret_cast<void*>(impl_->device) : nullptr;
}

void* Context::vulkan_instance() {
    return impl_ ? reinterpret_cast<void*>(impl_->instance) : nullptr;
}

void* Context::vulkan_physical_device() {
    return impl_ ? reinterpret_cast<void*>(impl_->physical_device) : nullptr;
}

uint32_t Context::compute_queue_family() const {
    return impl_ ? impl_->compute_queue_family : 0;
}

void* Context::compute_queue() {
    return impl_ ? reinterpret_cast<void*>(impl_->compute_queue) : nullptr;
}

} // namespace cw
