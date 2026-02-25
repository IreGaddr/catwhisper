#include <gtest/gtest.h>
#include <catwhisper/index_hnsw.hpp>
#include <catwhisper/index_flat.hpp>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Phase 4 IndexHNSW benchmarks
// Measures graph-based index performance and recall

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

float compute_recall(const std::vector<std::pair<float, uint64_t>>& ground_truth,
                     const SearchResults& results, uint32_t k) {
    std::set<uint64_t> gt_ids;
    for (uint32_t i = 0; i < std::min(k, static_cast<uint32_t>(ground_truth.size())); ++i) {
        gt_ids.insert(ground_truth[i].second);
    }

    auto query_results = results[0];
    uint32_t hits = 0;
    for (uint32_t i = 0; i < std::min(k, static_cast<uint32_t>(query_results.size())); ++i) {
        if (gt_ids.count(query_results[i].id)) {
            ++hits;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

std::vector<std::pair<float, uint64_t>> brute_force_search(
    const std::vector<float>& database,
    const std::vector<float>& query,
    uint64_t n_vectors, uint32_t dim, uint32_t k) {

    std::vector<std::pair<float, uint64_t>> results;
    results.reserve(n_vectors);

    for (uint64_t i = 0; i < n_vectors; ++i) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float diff = database[i * dim + d] - query[d];
            dist += diff * diff;
        }
        results.emplace_back(dist, i);
    }

    std::partial_sort(results.begin(), results.begin() + std::min(k, static_cast<uint32_t>(n_vectors)), results.end());
    results.resize(std::min(k, static_cast<uint32_t>(n_vectors)));
    return results;
}

class HNSWBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
    }

    static constexpr int N_WARMUP = 5;
    static constexpr int N_TIMED = 20;
};

// ============================================================================
// 10K x128: Compare HNSW vs IndexFlat baseline
// ============================================================================
TEST_F(HNSWBenchmark, Compare_10Kx128) {
    constexpr uint64_t N = 10'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;

    HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;

    auto data = generate_vectors(N, DIM, 42);
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);

    // --- IndexHNSW ---
    auto hnsw_result = IndexHNSW::create(DIM, params);
    ASSERT_TRUE(hnsw_result.has_value()) << hnsw_result.error().message();
    auto hnsw = std::move(*hnsw_result);

    auto add_result = hnsw.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    hnsw.set_ef_search(50);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = hnsw.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> hnsw_times;
    hnsw_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = hnsw.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        hnsw_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double hnsw_median = median_latency_ms(hnsw_times);
    double hnsw_mean = std::accumulate(hnsw_times.begin(), hnsw_times.end(), 0.0) / hnsw_times.size();

    // --- Recall check ---
    float total_recall = 0.0f;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> query(queries.begin() + (N_WARMUP + i) * DIM,
                                  queries.begin() + (N_WARMUP + i + 1) * DIM);
        auto gt = brute_force_search(data, query, N, DIM, K);
        auto result = hnsw.search(query, K);
        ASSERT_TRUE(result.has_value());
        total_recall += compute_recall(gt, *result, K);
    }
    float avg_recall = total_recall / 10.0f;

    RecordProperty("hnsw_median_ms", std::to_string(hnsw_median));
    RecordProperty("hnsw_mean_ms", std::to_string(hnsw_mean));
    RecordProperty("avg_recall", std::to_string(avg_recall));

    printf("\n");
    printf("  10K x128, M=%u, ef_construction=%u, ef_search=%u:\n",
           params.M, params.ef_construction, hnsw.ef_search());
    printf("    IndexHNSW: median %.3f ms, mean %.3f ms, QPS %.0f\n",
           hnsw_median, hnsw_mean, 1000.0 / hnsw_mean);
    printf("    Recall@%u: %.1f%%\n", K, avg_recall * 100.0f);

    EXPECT_GT(avg_recall, 0.70f) << "Recall too low";
}

// ============================================================================
// 100K x128: Larger scale HNSW test
// ============================================================================
TEST_F(HNSWBenchmark, Compare_100Kx128) {
    constexpr uint64_t N = 100'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;

    HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;

    auto data = generate_vectors(N, DIM, 42);
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);

    // --- IndexHNSW ---
    auto hnsw_result = IndexHNSW::create(DIM, params);
    ASSERT_TRUE(hnsw_result.has_value()) << hnsw_result.error().message();
    auto hnsw = std::move(*hnsw_result);

    auto add_result = hnsw.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    hnsw.set_ef_search(100);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = hnsw.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> hnsw_times;
    hnsw_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = hnsw.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        hnsw_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double hnsw_median = median_latency_ms(hnsw_times);
    double hnsw_mean = std::accumulate(hnsw_times.begin(), hnsw_times.end(), 0.0) / hnsw_times.size();

    // --- Recall check ---
    float total_recall = 0.0f;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> query(queries.begin() + (N_WARMUP + i) * DIM,
                                  queries.begin() + (N_WARMUP + i + 1) * DIM);
        auto gt = brute_force_search(data, query, N, DIM, K);
        auto result = hnsw.search(query, K);
        ASSERT_TRUE(result.has_value());
        total_recall += compute_recall(gt, *result, K);
    }
    float avg_recall = total_recall / 10.0f;

    RecordProperty("hnsw_median_ms", std::to_string(hnsw_median));
    RecordProperty("hnsw_mean_ms", std::to_string(hnsw_mean));
    RecordProperty("avg_recall", std::to_string(avg_recall));

    printf("\n");
    printf("  100K x128, M=%u, ef_construction=%u, ef_search=%u:\n",
           params.M, params.ef_construction, hnsw.ef_search());
    printf("    IndexHNSW: median %.3f ms, mean %.3f ms, QPS %.0f\n",
           hnsw_median, hnsw_mean, 1000.0 / hnsw_mean);
    printf("    Recall@%u: %.1f%%\n", K, avg_recall * 100.0f);

    EXPECT_GT(avg_recall, 0.70f) << "Recall too low";
}

// ============================================================================
// Recall sweep: Test different ef_search values
// ============================================================================
TEST_F(HNSWBenchmark, RecallSweep_10Kx128) {
    constexpr uint64_t N = 10'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;

    HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;

    auto data = generate_vectors(N, DIM, 42);
    auto queries = generate_vectors(10, DIM, 123);

    auto hnsw_result = IndexHNSW::create(DIM, params);
    ASSERT_TRUE(hnsw_result.has_value()) << hnsw_result.error().message();
    auto hnsw = std::move(*hnsw_result);

    auto add_result = hnsw.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    printf("\n  Recall sweep (10K x128, M=%u):\n", params.M);
    printf("    %-12s %-10s %-12s\n", "ef_search", "recall@10", "median ms");

    for (uint32_t ef_search : {10, 20, 50, 100, 200}) {
        hnsw.set_ef_search(ef_search);

        float total_recall = 0.0f;
        for (int i = 0; i < 10; ++i) {
            std::vector<float> query(queries.begin() + i * DIM,
                                      queries.begin() + (i + 1) * DIM);
            auto gt = brute_force_search(data, query, N, DIM, K);
            auto result = hnsw.search(query, K);
            ASSERT_TRUE(result.has_value());
            total_recall += compute_recall(gt, *result, K);
        }
        float avg_recall = total_recall / 10.0f;

        std::vector<double> times;
        for (int i = 0; i < 20; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto result = hnsw.search({queries.data() + (i % 10) * DIM, DIM}, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double median = median_latency_ms(times);

        printf("    %-12u %-10.1f %-12.3f\n", ef_search, avg_recall * 100.0f, median);
    }
}

} // namespace
