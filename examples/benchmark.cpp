#include <catwhisper/catwhisper.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [n_vectors] [dimension] [k]\n";
    std::cout << "  n_vectors: Number of vectors to index (default: 100000)\n";
    std::cout << "  dimension: Vector dimension (default: 128)\n";
    std::cout << "  k: Number of nearest neighbors (default: 10)\n";
}

int main(int argc, char* argv[]) {
    namespace cw = catwhisper;
    
    uint64_t n_vectors = 100000;
    uint32_t dimension = 128;
    uint32_t k = 10;
    
    if (argc > 1) {
        n_vectors = std::stoull(argv[1]);
    }
    if (argc > 2) {
        dimension = std::stoul(argv[2]);
    }
    if (argc > 3) {
        k = std::stoul(argv[3]);
    }
    
    std::cout << "CatWhisper Benchmark\n";
    std::cout << "====================\n";
    std::cout << "Vectors: " << n_vectors << "\n";
    std::cout << "Dimension: " << dimension << "\n";
    std::cout << "K: " << k << "\n\n";
    
    // Create context
    auto ctx_result = cw::Context::create();
    if (!ctx_result) {
        std::cerr << "Failed to create context: " << ctx_result.error().message() << "\n";
        return 1;
    }
    auto ctx = std::move(*ctx_result);
    
    std::cout << "Device: " << ctx.device_info().name << "\n\n";
    
    // Generate data
    std::cout << "Generating data...\n";
    std::vector<float> data(n_vectors * dimension);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) {
        v = dist(rng);
    }
    
    // Create index
    auto index_result = cw::IndexFlat::create(ctx, dimension);
    if (!index_result) {
        std::cerr << "Failed to create index: " << index_result.error().message() << "\n";
        return 1;
    }
    auto index = std::move(*index_result);
    
    // Add vectors
    std::cout << "Adding vectors...\n";
    auto add_start = std::chrono::high_resolution_clock::now();
    auto add_result = index.add(data, n_vectors);
    auto add_end = std::chrono::high_resolution_clock::now();
    
    if (!add_result) {
        std::cerr << "Failed to add: " << add_result.error().message() << "\n";
        return 1;
    }
    
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start).count();
    double add_throughput = (n_vectors * 1000.0) / add_ms;
    
    std::cout << "Add time: " << add_ms << "ms\n";
    std::cout << "Add throughput: " << add_throughput << " vectors/sec\n\n";
    
    // Warmup search
    std::vector<float> warmup_query(dimension);
    index.search(warmup_query, k);
    
    // Benchmark search
    const int n_queries = 100;
    std::vector<std::vector<float>> queries(n_queries);
    for (int i = 0; i < n_queries; ++i) {
        queries[i].resize(dimension);
        for (uint32_t j = 0; j < dimension; ++j) {
            queries[i][j] = dist(rng);
        }
    }
    
    std::cout << "Running " << n_queries << " searches...\n";
    
    std::vector<double> search_times;
    search_times.reserve(n_queries);
    
    for (int i = 0; i < n_queries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = index.search(queries[i], k);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!result) {
            std::cerr << "Search failed\n";
            return 1;
        }
        
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        search_times.push_back(us / 1000.0);
    }
    
    std::sort(search_times.begin(), search_times.end());
    
    double mean = 0;
    for (double t : search_times) mean += t;
    mean /= n_queries;
    
    double median = search_times[n_queries / 2];
    double p95 = search_times[static_cast<int>(n_queries * 0.95)];
    double p99 = search_times[static_cast<int>(n_queries * 0.99)];
    
    std::cout << "Search latency (ms):\n";
    std::cout << "  Mean:   " << mean << "\n";
    std::cout << "  Median: " << median << "\n";
    std::cout << "  P95:    " << p95 << "\n";
    std::cout << "  P99:    " << p99 << "\n";
    std::cout << "  Throughput: " << (1000.0 / mean) << " queries/sec\n";
    
    return 0;
}
