#include <catwhisper/catwhisper.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    namespace cw = catwhisper;
    
    std::cout << "=== CatWhisper Basic Example ===\n\n";
    
    // List available devices
    auto devices = cw::Context::list_devices();
    if (!devices) {
        std::cerr << "Failed to list devices: " << devices.error().message() << "\n";
        return 1;
    }
    
    std::cout << "Available GPU devices:\n";
    for (const auto& dev : *devices) {
        std::cout << "  [" << dev.device_id << "] " << dev.name 
                  << " (" << (dev.total_memory / (1024 * 1024)) << " MB)\n";
    }
    std::cout << "\n";
    
    // Create context
    std::cout << "Creating Vulkan context...\n";
    auto ctx_result = cw::Context::create();
    if (!ctx_result) {
        std::cerr << "Failed to create context: " << ctx_result.error().message() << "\n";
        return 1;
    }
    auto ctx = std::move(*ctx_result);
    
    const auto& info = ctx.device_info();
    std::cout << "Using device: " << info.name << "\n";
    std::cout << "  Total memory: " << (info.total_memory / (1024 * 1024)) << " MB\n";
    std::cout << "  Subgroup size: " << info.subgroup_size << "\n";
    std::cout << "  Max workgroup: " << info.max_workgroup_size << "\n\n";
    
    // Create index
    const uint32_t dim = 128;
    const uint64_t n_vectors = 10000;
    
    std::cout << "Creating IndexFlat (dim=" << dim << ")...\n";
    auto index_result = cw::IndexFlat::create(ctx, dim);
    if (!index_result) {
        std::cerr << "Failed to create index: " << index_result.error().message() << "\n";
        return 1;
    }
    auto index = std::move(*index_result);
    
    // Generate random data
    std::cout << "Generating " << n_vectors << " random vectors...\n";
    std::vector<float> data(n_vectors * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : data) {
        v = dist(rng);
    }
    
    // Add vectors to index
    std::cout << "Adding vectors to index...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto add_result = index.add(data, n_vectors);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!add_result) {
        std::cerr << "Failed to add vectors: " << add_result.error().message() << "\n";
        return 1;
    }
    
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Added " << n_vectors << " vectors in " << add_ms << "ms\n\n";
    
    // Search
    const uint32_t k = 10;
    std::cout << "Searching for nearest " << k << " neighbors...\n";
    
    std::vector<float> query(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        query[i] = data[i];  // Use first vector as query
    }
    
    start = std::chrono::high_resolution_clock::now();
    auto search_result = index.search(query, k);
    end = std::chrono::high_resolution_clock::now();
    
    if (!search_result) {
        std::cerr << "Search failed: " << search_result.error().message() << "\n";
        return 1;
    }
    
    auto search_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Search completed in " << search_us << "us\n\n";
    
    // Display results
    std::cout << "Top " << k << " results:\n";
    auto results = (*search_result)[0];
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i + 1) << ". ID=" << results[i].id 
                  << ", distance=" << results[i].distance << "\n";
    }
    
    // Verify first result is the query itself (distance should be ~0)
    if (!results.empty() && results[0].id == 0) {
        std::cout << "\nVerification: First result is the query vector (distance ≈ 0)\n";
    }
    
    std::cout << "\nDone!\n";
    return 0;
}
