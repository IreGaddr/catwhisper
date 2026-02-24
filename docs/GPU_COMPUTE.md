# GPU Compute Pipeline Design

## Why Vulkan Compute?

Vulkan compute provides:

1. **Cross-vendor support**: Works on NVIDIA, AMD, Intel, ARM, Apple (via MoltenVK)
2. **Low-level control**: Direct memory management, explicit synchronization
3. **No runtime overhead**: Unlike CUDA's driver model
4. **Future-proof**: Industry standard, widely supported

The trade-off is more complex code compared to CUDA, but the portability is worth it.

## Vulkan Initialization

```cpp
class VulkanContext {
    // Instance creation with validation layers in debug
    void createInstance() {
        VkApplicationInfo app_info = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "CatWhisper",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = VK_API_VERSION_1_3
        };
        
        std::vector<const char*> extensions = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME
        };
        
        // Enable validation in debug builds
        std::vector<const char*> layers;
        #ifdef CW_DEBUG
        layers.push_back("VK_LAYER_KHRONOS_validation");
        #endif
        
        VkInstanceCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = layers.size(),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = extensions.size(),
            .ppEnabledExtensionNames = extensions.data()
        };
        
        vkCreateInstance(&create_info, nullptr, &instance_);
    }
    
    // Select best GPU (prefer discrete, high VRAM)
    void selectPhysicalDevice() {
        uint32_t device_count;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
        
        // Score devices
        int best_score = -1;
        for (auto device : devices) {
            int score = scoreDevice(device);
            if (score > best_score) {
                best_score = score;
                physical_device_ = device;
            }
        }
    }
    
    int scoreDevice(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        
        int score = 0;
        
        // Prefer discrete GPUs
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 1000;
        
        // Prefer more VRAM
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(device, &mem_props);
        
        VkDeviceSize total_memory = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                total_memory += mem_props.memoryHeaps[i].size;
        }
        score += total_memory / (1024 * 1024);  // MB
        
        // Check compute queue support
        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
        
        bool has_compute = false;
        for (const auto& family : queue_families) {
            if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute = true;
                break;
            }
        }
        
        if (!has_compute) return -1;
        
        return score;
    }
};
```

## Compute Queue Setup

```cpp
void createDeviceAndQueues() {
    // Find compute queue family
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
    
    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_families.size(); ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    
    // Enable useful extensions
    std::vector<const char*> extensions = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
    };
    
    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
    };
    features2.features.shaderFloat64 = VK_TRUE;
    features2.features.shaderInt16 = VK_TRUE;
    features2.features.shaderInt64 = VK_TRUE;
    
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = extensions.size(),
        .ppEnabledExtensionNames = extensions.data()
    };
    
    vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    
    vkGetDeviceQueue(device_, compute_queue_family, 0, &compute_queue_);
    compute_queue_family_ = compute_queue_family;
}
```

## Memory Management with VMA

```cpp
#include <vk_mem_alloc.h>

class GPUMemoryAllocator {
    VmaAllocator allocator_;
    
public:
    void init(VkInstance instance, VkPhysicalDevice phys_dev, VkDevice device) {
        VmaVulkanFunctions vulkan_functions = {
            .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr = vkGetDeviceProcAddr
        };
        
        VmaAllocatorCreateInfo create_info = {
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = phys_dev,
            .device = device,
            .pVulkanFunctions = &vulkan_functions,
            .instance = instance,
            .vulkanApiVersion = VK_API_VERSION_1_3
        };
        
        vmaCreateAllocator(&create_info, &allocator_);
    }
    
    struct Buffer {
        VkBuffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo info;
        void* mapped = nullptr;
    };
    
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                        VmaMemoryUsage memory_usage, bool map = false) {
        VkBufferCreateInfo buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage
        };
        
        VmaAllocationCreateInfo alloc_info = {
            .usage = memory_usage
        };
        
        Buffer result;
        vmaCreateBuffer(allocator_, &buffer_info, &alloc_info,
                        &result.buffer, &result.allocation, &result.info);
        
        if (map) {
            vmaMapMemory(allocator_, result.allocation, &result.mapped);
        }
        
        return result;
    }
    
    void destroyBuffer(Buffer& buffer) {
        if (buffer.mapped) {
            vmaUnmapMemory(allocator_, buffer.allocation);
        }
        vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
    }
};
```

## Shader Hotloading (Development)

For rapid iteration, we compile GLSL to SPIR-V at runtime in debug builds:

```cpp
class ShaderManager {
    std::filesystem::path shader_dir_;
    std::unordered_map<std::string, VkShaderModule> cache_;
    
public:
    VkShaderModule getShader(VkDevice device, const std::string& name) {
        auto it = cache_.find(name);
        if (it != cache_.end()) {
            return it->second;
        }
        
        // Compile GLSL to SPIR-V
        std::vector<uint32_t> spirv = compileGLSL(name);
        
        VkShaderModuleCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.size() * sizeof(uint32_t),
            .pCode = spirv.data()
        };
        
        VkShaderModule module;
        vkCreateShaderModule(device, &create_info, nullptr, &module);
        
        cache_[name] = module;
        return module;
    }
    
    std::vector<uint32_t> compileGLSL(const std::string& name) {
        std::string glsl_path = (shader_dir_ / (name + ".comp")).string();
        std::string spirv_path = (shader_dir_ / (name + ".spv")).string();
        
        // Use glslangValidator or shaderc
        std::string cmd = "glslangValidator -V " + glsl_path + " -o " + spirv_path;
        system(cmd.c_str());
        
        // Read SPIR-V
        std::ifstream file(spirv_path, std::ios::binary);
        std::vector<uint32_t> spirv((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
        return spirv;
    }
};
```

## Core Compute Shaders

### Distance Computation (L2)

```glsl
// distance_l2.comp
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer DatabaseBuffer {
    float16_t database[];
};

layout(set = 0, binding = 1) readonly buffer QueryBuffer {
    float16_t query[];
};

layout(set = 0, binding = 2) writeonly buffer DistanceBuffer {
    float distances[];
};

layout(push_constant) uniform PushConstants {
    uint n_vectors;      // Number of database vectors
    uint dimension;      // Vector dimension
    uint query_idx;      // Which query vector
    uint pad;
};

shared float16_t shared_query[256];  // Assuming dim <= 256

void main() {
    uint vec_idx = gl_GlobalInvocationID.x;
    uint dim_idx = gl_LocalInvocationID.x;
    
    // Load query vector into shared memory (cooperative)
    if (gl_LocalInvocationID.x < dimension) {
        shared_query[dim_idx] = query[query_idx * dimension + dim_idx];
    }
    barrier();
    
    if (vec_idx >= n_vectors) return;
    
    // Compute L2 distance using subgroup operations
    float local_sum = 0.0;
    for (uint i = dim_idx; i < dimension; i += gl_WorkGroupSize.x) {
        float diff = float(database[vec_idx * dimension + i]) - float(shared_query[i]);
        local_sum += diff * diff;
    }
    
    // Subgroup reduction
    float subgroup_sum = subgroupAdd(local_sum);
    
    // First thread in subgroup writes result
    if (subgroupElect()) {
        distances[vec_idx] = subgroup_sum;
    }
}
```

### Top-K Selection

```glsl
// topk_select.comp
#version 450
#extension GL_KHR_shader_subgroup_shuffle : enable

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer DistanceBuffer {
    float distances[];
};

layout(set = 0, binding = 1) writeonly buffer ResultBuffer {
    uint result_indices[];
    float result_distances[];
};

layout(push_constant) uniform PushConstants {
    uint n_vectors;
    uint k;
};

// Heap-based top-k in shared memory
shared uint heap_indices[256];
shared float heap_distances[256];

void heapifyUp(int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap_distances[idx] < heap_distances[parent]) {
            // Min-heap: swap if smaller
            float tmp_dist = heap_distances[idx];
            heap_distances[idx] = heap_distances[parent];
            heap_distances[parent] = tmp_dist;
            
            uint tmp_idx = heap_indices[idx];
            heap_indices[idx] = heap_indices[parent];
            heap_indices[parent] = tmp_idx;
            
            idx = parent;
        } else {
            break;
        }
    }
}

void heapReplace(uint new_idx, float new_dist) {
    if (new_dist > heap_distances[0]) {
        // Replace root with larger element
        heap_distances[0] = new_dist;
        heap_indices[0] = new_idx;
        
        // Heapify down
        int idx = 0;
        while (true) {
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            int smallest = idx;
            
            if (left < k && heap_distances[left] < heap_distances[smallest])
                smallest = left;
            if (right < k && heap_distances[right] < heap_distances[smallest])
                smallest = right;
            
            if (smallest != idx) {
                float tmp_dist = heap_distances[idx];
                heap_distances[idx] = heap_distances[smallest];
                heap_distances[smallest] = tmp_dist;
                
                uint tmp_idx = heap_indices[idx];
                heap_indices[idx] = heap_indices[smallest];
                heap_indices[smallest] = tmp_idx;
                
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    
    // Initialize heap with first k elements (thread 0..k-1)
    if (gid < k) {
        heap_indices[tid] = gid;
        heap_distances[tid] = distances[gid];
    } else {
        heap_indices[tid] = 0;
        heap_distances[tid] = 1.0 / 0.0;  // infinity
    }
    barrier();
    
    // Build initial heap (single thread)
    if (tid == 0) {
        for (int i = 1; i < int(k) && i < int(n_vectors); ++i) {
            heapifyUp(i);
        }
    }
    barrier();
    
    // Process remaining elements
    for (uint i = gid + k; i < n_vectors; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {
        float dist = distances[i];
        heapReplace(i, dist);
        barrier();  // Need barrier after each modification
    }
    
    // Write results (only first k threads)
    if (tid < k) {
        result_indices[tid] = heap_indices[tid];
        result_distances[tid] = heap_distances[tid];
    }
}
```

## Pipeline Creation

```cpp
class ComputePipeline {
    VkPipeline pipeline_;
    VkPipelineLayout layout_;
    VkDescriptorSetLayout set_layout_;
    
public:
    void create(VkDevice device, VkShaderModule shader, 
                uint32_t push_constant_size) {
        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };
        
        VkDescriptorSetLayoutCreateInfo set_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = bindings.size(),
            .pBindings = bindings.data()
        };
        
        vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout_);
        
        // Pipeline layout
        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_constant_size
        };
        
        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range
        };
        
        vkCreatePipelineLayout(device, &layout_info, nullptr, &layout_);
        
        // Compute pipeline
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main"
            },
            .layout = layout_
        };
        
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, 
                                 &pipeline_info, nullptr, &pipeline_);
    }
};
```

## Command Buffer Recording

```cpp
class CommandRecorder {
    VkCommandPool pool_;
    VkDevice device_;
    
public:
    VkCommandBuffer recordSearch(IndexFlat& index, 
                                  const float* queries, 
                                  uint32_t n_queries,
                                  uint32_t k) {
        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        vkAllocateCommandBuffers(device_, &alloc_info, &cmd);
        
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(cmd, &begin_info);
        
        // Barrier: ensure query data is visible
        VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        // For each query
        for (uint32_t q = 0; q < n_queries; ++q) {
            // Dispatch distance computation
            struct PushConstants {
                uint32_t n_vectors;
                uint32_t dimension;
                uint32_t query_idx;
                uint32_t pad;
            } pc = { index.size(), index.dimension(), q, 0 };
            
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                             index.distance_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   index.pipeline_layout_, 0, 1, 
                                   &index.descriptor_set_, 0, nullptr);
            vkCmdPushConstants(cmd, index.pipeline_layout_,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            
            uint32_t groups = (index.size() + 255) / 256;
            vkCmdDispatch(cmd, groups, 1, 1);
            
            // Barrier before top-k
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
            
            // Dispatch top-k selection
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             index.topk_pipeline_);
            struct TopKPushConstants {
                uint32_t n_vectors;
                uint32_t k;
            } topk_pc = { index.size(), k };
            vkCmdPushConstants(cmd, index.topk_pipeline_layout_,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, 
                              sizeof(topk_pc), &topk_pc);
            vkCmdDispatch(cmd, 1, 1, 1);
        }
        
        vkEndCommandBuffer(cmd);
        return cmd;
    }
};
```

## Subgroup Optimization

Modern GPUs support "subgroup operations" (NVIDIA: warps, AMD: wavefronts). These are crucial for performance:

```cpp
void checkSubgroupCapabilities(VkPhysicalDevice device) {
    VkPhysicalDeviceSubgroupProperties subgroup_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
    };
    
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props
    };
    
    vkGetPhysicalDeviceProperties2(device, &props2);
    
    // Check we have what we need
    assert(subgroup_props.subgroupSize >= 32);  // At least 32 threads
    assert(subgroup_props.supportedOperations & 
           VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);  // subgroupAdd, etc.
    assert(subgroup_props.supportedStages & 
           VK_SHADER_STAGE_COMPUTE_BIT);
}
```

## Performance Considerations

### Memory Coalescing

Ensure adjacent threads access adjacent memory:

```glsl
// GOOD: Coalesced (adjacent threads read adjacent memory)
float val = database[vec_idx * dimension + gl_LocalInvocationID.x];

// BAD: Strided (causes cache thrashing)
float val = database[gl_LocalInvocationID.x * dimension + vec_idx];
```

### Bank Conflicts

Avoid shared memory bank conflicts:

```glsl
// Pad shared memory to avoid bank conflicts
shared float data[256 + 16];  // 16 floats of padding
```

### Occupancy

Balance work per thread with register pressure:

```cpp
// Query device limits
VkPhysicalDeviceComputeShaderProperties compute_props;
// ...
uint32_t max_workgroup_size = compute_props.maxComputeWorkGroupSize[0];
uint32_t max_shared_memory = compute_props.maxComputeSharedMemorySize;

// Choose workgroup size based on kernel requirements
```

### Async Compute

Overlap compute with data transfer:

```cpp
// Create separate queue for transfers
VkQueue transfer_queue;

// Use fences for synchronization
VkFence fence;
VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
vkCreateFence(device, &fence_info, nullptr, &fence);

// Submit compute work
vkQueueSubmit(compute_queue, 1, &submit_info, fence);

// Can do other work here...

// Wait for completion
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
```
