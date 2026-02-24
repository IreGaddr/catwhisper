#ifndef CATWHISPER_CONTEXT_IMPL_HPP
#define CATWHISPER_CONTEXT_IMPL_HPP

#include <catwhisper/types.hpp>
#include <catwhisper/error.hpp>
#include <catwhisper/context.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <cstring>

namespace cw {

struct Context::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    
    uint32_t compute_queue_family = 0;
    DeviceInfo device_info;
    
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    
    Expected<void> init(const ContextOptions& options) {
        if (auto r = create_instance(options); !r) {
            return make_unexpected(r.error().code(), r.error().message());
        }
        
        if (auto r = pick_physical_device(options); !r) {
            return make_unexpected(r.error().code(), r.error().message());
        }
        
        if (auto r = create_device(options); !r) {
            return make_unexpected(r.error().code(), r.error().message());
        }
        
        if (auto r = create_command_pool(); !r) {
            return make_unexpected(r.error().code(), r.error().message());
        }
        
        if (auto r = create_allocator(); !r) {
            return make_unexpected(r.error().code(), r.error().message());
        }
        
        return {};
    }
    
    static Expected<std::vector<DeviceInfo>> list_devices() {
        std::vector<DeviceInfo> devices;
        
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "CatWhisper Device Enum";
        app_info.applicationVersion = 0;
        app_info.apiVersion = VK_API_VERSION_1_3;
        
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        
        VkInstance temp_instance;
        if (vkCreateInstance(&create_info, nullptr, &temp_instance) != VK_SUCCESS) {
            return make_unexpected(ErrorCode::InstanceCreationFailed, 
                                   "Failed to create temporary Vulkan instance");
        }
        
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(temp_instance, &device_count, nullptr);
        std::vector<VkPhysicalDevice> phys_devices(device_count);
        vkEnumeratePhysicalDevices(temp_instance, &device_count, phys_devices.data());
        
        for (uint32_t i = 0; i < device_count; ++i) {
            DeviceInfo info = get_device_info(phys_devices[i], i);
            devices.push_back(info);
        }
        
        vkDestroyInstance(temp_instance, nullptr);
        return devices;
    }
    
    void destroy() {
        if (allocator) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
        if (command_pool) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        if (device) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (debug_messenger) {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
            if (func) {
                func(instance, debug_messenger, nullptr);
            }
            debug_messenger = VK_NULL_HANDLE;
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }
    
    ~Impl() {
        destroy();
    }
    
    uint64_t available_memory() const {
        // Simplified: just return total memory
        return device_info.total_memory;
    }
    
    void synchronize() {
        vkDeviceWaitIdle(device);
    }

private:
    Expected<void> create_instance(const ContextOptions& options) {
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "CatWhisper";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;
        
        std::vector<const char*> extensions;
        std::vector<const char*> layers;
        
#ifdef CW_VULKAN_VALIDATION
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        // Try to use validation layer, but don't fail if unavailable
        layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
        
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        
        VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
        if (result != VK_SUCCESS) {
            // Retry without validation layers if they're not available
            if (!layers.empty()) {
                create_info.enabledLayerCount = 0;
                create_info.ppEnabledLayerNames = nullptr;
                result = vkCreateInstance(&create_info, nullptr, &instance);
            }
            if (result != VK_SUCCESS) {
                return make_unexpected(ErrorCode::InstanceCreationFailed,
                                       "Failed to create Vulkan instance");
            }
        }
        
#ifdef CW_VULKAN_VALIDATION
        setup_debug_messenger();
#endif
        
        return {};
    }
    
    Expected<void> pick_physical_device(const ContextOptions& options) {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        
        if (device_count == 0) {
            return make_unexpected(ErrorCode::NoComputeCapableDevice,
                                   "No Vulkan-capable devices found");
        }
        
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        
        int best_score = -1;
        uint32_t best_idx = 0;
        
        for (uint32_t i = 0; i < device_count; ++i) {
            int score = score_device(devices[i]);
            if (options.device_id >= 0 && static_cast<int>(i) == options.device_id) {
                physical_device = devices[i];
                device_info = get_device_info(physical_device, i);
                return {};
            }
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        
        if (best_score < 0) {
            return make_unexpected(ErrorCode::NoComputeCapableDevice,
                                   "No compute-capable GPU found");
        }
        
        physical_device = devices[best_idx];
        device_info = get_device_info(physical_device, best_idx);
        return {};
    }
    
    Expected<void> create_device(const ContextOptions& options) {
        (void)options;  // Suppress unused warning
        
        compute_queue_family = find_compute_queue_family();
        
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info = {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = compute_queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        
        VkPhysicalDeviceFeatures features = {};
        features.shaderFloat64 = VK_TRUE;
        features.shaderInt16 = VK_TRUE;
        
        VkDeviceCreateInfo device_info_create = {};
        device_info_create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info_create.queueCreateInfoCount = 1;
        device_info_create.pQueueCreateInfos = &queue_info;
        device_info_create.pEnabledFeatures = &features;
        
        if (vkCreateDevice(physical_device, &device_info_create, nullptr, &device) != VK_SUCCESS) {
            return make_unexpected(ErrorCode::DeviceCreationFailed,
                                   "Failed to create Vulkan device");
        }
        
        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);
        return {};
    }
    
    Expected<void> create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family;
        
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            return make_unexpected(ErrorCode::OperationFailed,
                                   "Failed to create command pool");
        }
        
        return {};
    }
    
    Expected<void> create_allocator() {
        VmaVulkanFunctions vulkan_functions = {};
        vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        
        VmaAllocatorCreateInfo create_info = {};
        create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        create_info.physicalDevice = physical_device;
        create_info.device = device;
        create_info.pVulkanFunctions = &vulkan_functions;
        create_info.instance = instance;
        create_info.vulkanApiVersion = VK_API_VERSION_1_3;
        
        if (vmaCreateAllocator(&create_info, &allocator) != VK_SUCCESS) {
            return make_unexpected(ErrorCode::AllocationFailed,
                                   "Failed to create VMA allocator");
        }
        
        return {};
    }
    
    uint32_t find_compute_queue_family() {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
        
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return i;
            }
        }
        
        return 0;
    }
    
    int score_device(VkPhysicalDevice dev) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        
        int score = 0;
        
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 500;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            score += 100;
        }
        
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(dev, &mem_props);
        
        uint64_t local_mem = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                local_mem += mem_props.memoryHeaps[i].size;
            }
        }
        score += static_cast<int>(local_mem / (1024 * 1024));
        
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, queue_families.data());
        
        bool has_compute = false;
        for (const auto& family : queue_families) {
            if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute = true;
                break;
            }
        }
        
        if (!has_compute) {
            return -1;
        }
        
        return score;
    }
    
    static DeviceInfo get_device_info(VkPhysicalDevice dev, uint32_t id) {
        DeviceInfo info;
        info.device_id = id;
        
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        
        info.name = props.deviceName;
        info.driver_version = std::to_string(props.driverVersion);
        info.max_workgroup_size = props.limits.maxComputeWorkGroupSize[0];
        
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(dev, &mem_props);
        
        info.total_memory = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                info.total_memory += mem_props.memoryHeaps[i].size;
            }
        }
        
        info.supports_fp16 = (props.apiVersion >= VK_API_VERSION_1_2);
        info.supports_int8 = (props.apiVersion >= VK_API_VERSION_1_2);
        info.subgroup_size = 32;  // Default subgroup size
        
        return info;
    }
    
#ifdef CW_VULKAN_VALIDATION
    void setup_debug_messenger() {
        VkDebugUtilsMessengerCreateInfoEXT create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debug_callback;
        
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func) {
            func(instance, &create_info, nullptr, &debug_messenger);
        }
    }
    
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user_data)
    {
        (void)type; (void)user_data;
        
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
        }
        
        return VK_FALSE;
    }
#endif
};

} // namespace cw

#endif // CATWHISPER_CONTEXT_IMPL_HPP
