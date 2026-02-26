#include <gtest/gtest.h>
#include <catwhisper/context.hpp>
#include <catwhisper/index_flat.hpp>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>

// ROTRTA: Ridiculous Optimizations Through Ridiculous Test Assertions
// AMD Ryzen 7 H 255 + Radeon 780M iGPU
//
// Baseline (current performance):
//   10K x128:  0.277 ms -> target 0.139 ms (50% reduction)
//   100K x128: 0.675 ms -> target 0.338 ms (50% reduction)  
//   100K x256: 0.937 ms -> target 0.469 ms (50% reduction)

namespace {

using namespace cw;

double median_latency_ms(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

std::vector<float> generate_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

class RotrtaAmdBenchmark : public ::testing::Test {
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
// 10K x128 k=10: Baseline 0.277 ms -> Target 0.139 ms (50% reduction)
// ============================================================================
TEST_F(RotrtaAmdBenchmark, Flat_10Kx128) {
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
    
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
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
    
    // ROTRTA: 50% reduction from 0.277 ms baseline
    EXPECT_LT(median, 0.139) 
        << "10Kx128 median " << median << " ms, target is 0.139 ms (50% reduction)";
}

// ============================================================================
// 100K x128 k=10: Baseline 0.675 ms -> Target 0.338 ms (50% reduction)
// ============================================================================
TEST_F(RotrtaAmdBenchmark, Flat_100Kx128) {
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
    
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
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
    
    // ROTRTA: 50% reduction from 0.675 ms baseline
    EXPECT_LT(median, 0.338)
        << "100Kx128 median " << median << " ms, target is 0.338 ms (50% reduction)";
}

// ============================================================================
// 100K x256 k=10: Baseline 0.937 ms -> Target 0.469 ms (50% reduction)
// ============================================================================
TEST_F(RotrtaAmdBenchmark, Flat_100Kx256) {
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
    
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = index.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }
    
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
    
    // ROTRTA: 50% reduction from 0.937 ms baseline
    EXPECT_LT(median, 0.469)
        << "100Kx256 median " << median << " ms, target is 0.469 ms (50% reduction)";
}

} // namespace
