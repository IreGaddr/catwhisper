#include <gtest/gtest.h>
#include <catwhisper/context.hpp>
#include <catwhisper/index_flat.hpp>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>

// ROTRTA: Ridiculous Optimizations Through Ridiculous Test Assertions
// These tests define the performance budget needed to BEAT FAISS-GPU.
//
// Current vs Target (single-query latency):
//   10K x128:  0.109 ms -> 0.064 ms (41% reduction needed)
//   100K x128: 0.416 ms -> 0.349 ms (16% reduction needed)  
//   100K x256: 1.198 ms -> 0.600 ms (50% reduction needed)
//
// FAISS-GPU reference (RTX 4080 Laptop):
//   10K x128:  0.065 ms
//   100K x128: 0.350 ms
//   100K x256: 0.601 ms

namespace {

using namespace cw;

// Timing helper: returns median of N runs in milliseconds
double median_latency_ms(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// Generate random vectors
std::vector<float> generate_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

class RotrtaBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx_result = Context::create();
        ASSERT_TRUE(ctx_result.has_value()) << ctx_result.error().message();
        ctx_ = std::make_unique<Context>(std::move(*ctx_result));
    }

    std::unique_ptr<Context> ctx_;
    static constexpr int N_WARMUP = 20;
    static constexpr int N_TIMED = 100;
};

// ============================================================================
// 10K x128 k=10: Target 0.064 ms (beat FAISS-GPU 0.065 ms)
// ============================================================================
TEST_F(RotrtaBenchmark, BeatsFaissGpu_10Kx128) {
    constexpr uint64_t N = 10'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;
    
    auto index_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(index_result.has_value()) << index_result.error().message();
    auto index = std::move(*index_result);
    
    auto data = generate_vectors(N, DIM);
    auto add_result = index.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();
    
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);
    
    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
    // Timed runs
    std::vector<double> times;
    times.reserve(N_TIMED);
    
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = index.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    
    double median = median_latency_ms(times);
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    RecordProperty("median_ms", std::to_string(median));
    RecordProperty("mean_ms", std::to_string(mean));
    
    // ROTRTA: Must beat FAISS-GPU 0.065 ms median
    EXPECT_LT(median, 0.065) 
        << "10Kx128 median " << median << " ms does not beat FAISS-GPU 0.065 ms";
}

// ============================================================================
// 100K x128 k=10: Target 0.349 ms (beat FAISS-GPU 0.350 ms)
// ============================================================================
TEST_F(RotrtaBenchmark, BeatsFaissGpu_100Kx128) {
    constexpr uint64_t N = 100'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;
    
    auto index_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(index_result.has_value()) << index_result.error().message();
    auto index = std::move(*index_result);
    
    auto data = generate_vectors(N, DIM);
    auto add_result = index.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();
    
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);
    
    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
    // Timed runs
    std::vector<double> times;
    times.reserve(N_TIMED);
    
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = index.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    
    double median = median_latency_ms(times);
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    RecordProperty("median_ms", std::to_string(median));
    RecordProperty("mean_ms", std::to_string(mean));
    
    // ROTRTA: Must beat FAISS-GPU 0.350 ms median
    EXPECT_LT(median, 0.349)
        << "100Kx128 median " << median << " ms does not beat FAISS-GPU 0.350 ms";
}

// ============================================================================
// 100K x256 k=10: Target 0.600 ms (beat FAISS-GPU 0.601 ms)
// ============================================================================
TEST_F(RotrtaBenchmark, BeatsFaissGpu_100Kx256) {
    constexpr uint64_t N = 100'000;
    constexpr uint32_t DIM = 256;
    constexpr uint32_t K = 10;
    
    auto index_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(index_result.has_value()) << index_result.error().message();
    auto index = std::move(*index_result);
    
    auto data = generate_vectors(N, DIM);
    auto add_result = index.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();
    
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);
    
    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
    // Timed runs
    std::vector<double> times;
    times.reserve(N_TIMED);
    
    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = index.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    
    double median = median_latency_ms(times);
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    RecordProperty("median_ms", std::to_string(median));
    RecordProperty("mean_ms", std::to_string(mean));
    
    // ROTRTA: Must beat FAISS-GPU 0.601 ms median
    EXPECT_LT(median, 0.600)
        << "100Kx256 median " << median << " ms does not beat FAISS-GPU 0.601 ms";
}

} // namespace
