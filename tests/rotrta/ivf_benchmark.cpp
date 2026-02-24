#include <gtest/gtest.h>
#include <catwhisper/context.hpp>
#include <catwhisper/index_ivf_flat.hpp>
#include <catwhisper/index_flat.hpp>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>

// Phase 2 IndexIVFFlat benchmarks
// Measures GPU-accelerated clustered index performance

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

// Generate clustered data for better IVF recall
std::vector<float> generate_clustered_vectors(uint64_t n, uint32_t dim, uint32_t n_clusters, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> cluster_center_dist(0.0f, 10.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.5f);
    std::uniform_int_distribution<uint32_t> cluster_dist(0, n_clusters - 1);

    // Generate cluster centers
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& center : centers) {
        center.resize(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            center[d] = cluster_center_dist(rng);
        }
    }

    // Generate vectors around cluster centers
    std::vector<float> data(n * dim);
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t c = cluster_dist(rng);
        for (uint32_t d = 0; d < dim; ++d) {
            data[i * dim + d] = centers[c][d] + noise_dist(rng);
        }
    }
    return data;
}

class IVFBenchmark : public ::testing::Test {
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

// Compute recall
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

// Brute force ground truth
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

// ============================================================================
// 10K x128: Compare IVF vs IndexFlat baseline
// ============================================================================
TEST_F(IVFBenchmark, Compare_10Kx128_nprobe16) {
    constexpr uint64_t N = 10'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;
    constexpr uint32_t NLIST = 32;
    constexpr uint32_t NPROBE = 16;

    auto data = generate_clustered_vectors(N, DIM, NLIST, 42);
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);

    // --- IndexIVFFlat ---
    IVFParams ivf_params{.nlist = NLIST, .nprobe = NPROBE, .kmeans_iters = 10};
    auto ivf_result = IndexIVFFlat::create(*ctx_, DIM, ivf_params);
    ASSERT_TRUE(ivf_result.has_value()) << ivf_result.error().message();
    auto ivf = std::move(*ivf_result);

    auto train_result = ivf.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = ivf.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = ivf.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> ivf_times;
    ivf_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = ivf.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        ivf_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double ivf_median = median_latency_ms(ivf_times);
    double ivf_mean = std::accumulate(ivf_times.begin(), ivf_times.end(), 0.0) / ivf_times.size();

    // --- IndexFlat baseline ---
    auto flat_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(flat_result.has_value()) << flat_result.error().message();
    auto flat = std::move(*flat_result);

    auto flat_add = flat.add(data, N);
    ASSERT_TRUE(flat_add.has_value());

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = flat.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> flat_times;
    flat_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = flat.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        flat_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double flat_median = median_latency_ms(flat_times);
    double flat_mean = std::accumulate(flat_times.begin(), flat_times.end(), 0.0) / flat_times.size();

    // --- Recall check ---
    float total_recall = 0.0f;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> query(queries.begin() + (N_WARMUP + i) * DIM,
                                  queries.begin() + (N_WARMUP + i + 1) * DIM);
        auto gt = brute_force_search(data, query, N, DIM, K);
        auto result = ivf.search(query, K);
        ASSERT_TRUE(result.has_value());
        total_recall += compute_recall(gt, *result, K);
    }
    float avg_recall = total_recall / 10.0f;

    RecordProperty("ivf_median_ms", std::to_string(ivf_median));
    RecordProperty("ivf_mean_ms", std::to_string(ivf_mean));
    RecordProperty("flat_median_ms", std::to_string(flat_median));
    RecordProperty("flat_mean_ms", std::to_string(flat_mean));
    RecordProperty("avg_recall", std::to_string(avg_recall));

    printf("\n");
    printf("  10K x128, nlist=%u, nprobe=%u:\n", NLIST, NPROBE);
    printf("    IndexIVFFlat: median %.3f ms, mean %.3f ms, QPS %.0f\n",
           ivf_median, ivf_mean, 1000.0 / ivf_mean);
    printf("    IndexFlat:    median %.3f ms, mean %.3f ms, QPS %.0f\n",
           flat_median, flat_mean, 1000.0 / flat_mean);
    printf("    Recall@%u:    %.1f%%\n", K, avg_recall * 100.0f);

    // Basic sanity checks
    EXPECT_GT(avg_recall, 0.7f) << "Recall too low";
}

// ============================================================================
// 100K x128: Larger scale IVF test
// ============================================================================
TEST_F(IVFBenchmark, Compare_100Kx128_nprobe16) {
    constexpr uint64_t N = 100'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;
    constexpr uint32_t NLIST = 64;
    constexpr uint32_t NPROBE = 16;

    auto data = generate_clustered_vectors(N, DIM, NLIST, 42);
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);

    // --- IndexIVFFlat ---
    IVFParams ivf_params{.nlist = NLIST, .nprobe = NPROBE, .kmeans_iters = 10};
    auto ivf_result = IndexIVFFlat::create(*ctx_, DIM, ivf_params);
    ASSERT_TRUE(ivf_result.has_value()) << ivf_result.error().message();
    auto ivf = std::move(*ivf_result);

    auto train_result = ivf.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = ivf.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = ivf.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> ivf_times;
    ivf_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = ivf.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        ivf_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double ivf_median = median_latency_ms(ivf_times);
    double ivf_mean = std::accumulate(ivf_times.begin(), ivf_times.end(), 0.0) / ivf_times.size();

    // --- IndexFlat baseline ---
    auto flat_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(flat_result.has_value()) << flat_result.error().message();
    auto flat = std::move(*flat_result);

    auto flat_add = flat.add(data, N);
    ASSERT_TRUE(flat_add.has_value());

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = flat.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> flat_times;
    flat_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = flat.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        flat_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double flat_median = median_latency_ms(flat_times);
    double flat_mean = std::accumulate(flat_times.begin(), flat_times.end(), 0.0) / flat_times.size();

    // --- Recall check ---
    float total_recall = 0.0f;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> query(queries.begin() + (N_WARMUP + i) * DIM,
                                  queries.begin() + (N_WARMUP + i + 1) * DIM);
        auto gt = brute_force_search(data, query, N, DIM, K);
        auto result = ivf.search(query, K);
        ASSERT_TRUE(result.has_value());
        total_recall += compute_recall(gt, *result, K);
    }
    float avg_recall = total_recall / 10.0f;

    RecordProperty("ivf_median_ms", std::to_string(ivf_median));
    RecordProperty("ivf_mean_ms", std::to_string(ivf_mean));
    RecordProperty("flat_median_ms", std::to_string(flat_median));
    RecordProperty("flat_mean_ms", std::to_string(flat_mean));
    RecordProperty("avg_recall", std::to_string(avg_recall));

    printf("\n");
    printf("  100K x128, nlist=%u, nprobe=%u:\n", NLIST, NPROBE);
    printf("    IndexIVFFlat: median %.3f ms, mean %.3f ms, QPS %.0f\n",
           ivf_median, ivf_mean, 1000.0 / ivf_mean);
    printf("    IndexFlat:    median %.3f ms, mean %.3f ms, QPS %.0f\n",
           flat_median, flat_mean, 1000.0 / flat_mean);
    printf("    Recall@%u:    %.1f%%\n", K, avg_recall * 100.0f);
    printf("    Speedup vs flat: %.2fx\n", flat_mean / ivf_mean);

    EXPECT_GT(avg_recall, 0.7f) << "Recall too low";
}

// ============================================================================
// 100K x256: Higher dimension test
// ============================================================================
TEST_F(IVFBenchmark, Compare_100Kx256_nprobe16) {
    constexpr uint64_t N = 100'000;
    constexpr uint32_t DIM = 256;
    constexpr uint32_t K = 10;
    constexpr uint32_t NLIST = 64;
    constexpr uint32_t NPROBE = 16;

    auto data = generate_clustered_vectors(N, DIM, NLIST, 42);
    auto queries = generate_vectors(N_WARMUP + N_TIMED, DIM, 123);

    // --- IndexIVFFlat ---
    IVFParams ivf_params{.nlist = NLIST, .nprobe = NPROBE, .kmeans_iters = 10};
    auto ivf_result = IndexIVFFlat::create(*ctx_, DIM, ivf_params);
    ASSERT_TRUE(ivf_result.has_value()) << ivf_result.error().message();
    auto ivf = std::move(*ivf_result);

    auto train_result = ivf.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = ivf.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = ivf.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> ivf_times;
    ivf_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = ivf.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        ivf_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double ivf_median = median_latency_ms(ivf_times);
    double ivf_mean = std::accumulate(ivf_times.begin(), ivf_times.end(), 0.0) / ivf_times.size();

    // --- IndexFlat baseline ---
    auto flat_result = IndexFlat::create(*ctx_, DIM);
    ASSERT_TRUE(flat_result.has_value()) << flat_result.error().message();
    auto flat = std::move(*flat_result);

    auto flat_add = flat.add(data, N);
    ASSERT_TRUE(flat_add.has_value());

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        auto result = flat.search({queries.data() + i * DIM, DIM}, K);
        ASSERT_TRUE(result.has_value());
    }

    // Timed runs
    std::vector<double> flat_times;
    flat_times.reserve(N_TIMED);

    for (int i = 0; i < N_TIMED; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = flat.search({queries.data() + (N_WARMUP + i) * DIM, DIM}, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(result.has_value());
        flat_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double flat_median = median_latency_ms(flat_times);
    double flat_mean = std::accumulate(flat_times.begin(), flat_times.end(), 0.0) / flat_times.size();

    // --- Recall check ---
    float total_recall = 0.0f;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> query(queries.begin() + (N_WARMUP + i) * DIM,
                                  queries.begin() + (N_WARMUP + i + 1) * DIM);
        auto gt = brute_force_search(data, query, N, DIM, K);
        auto result = ivf.search(query, K);
        ASSERT_TRUE(result.has_value());
        total_recall += compute_recall(gt, *result, K);
    }
    float avg_recall = total_recall / 10.0f;

    RecordProperty("ivf_median_ms", std::to_string(ivf_median));
    RecordProperty("ivf_mean_ms", std::to_string(ivf_mean));
    RecordProperty("flat_median_ms", std::to_string(flat_median));
    RecordProperty("flat_mean_ms", std::to_string(flat_mean));
    RecordProperty("avg_recall", std::to_string(avg_recall));

    printf("\n");
    printf("  100K x256, nlist=%u, nprobe=%u:\n", NLIST, NPROBE);
    printf("    IndexIVFFlat: median %.3f ms, mean %.3f ms, QPS %.0f\n",
           ivf_median, ivf_mean, 1000.0 / ivf_mean);
    printf("    IndexFlat:    median %.3f ms, mean %.3f ms, QPS %.0f\n",
           flat_median, flat_mean, 1000.0 / flat_mean);
    printf("    Recall@%u:    %.1f%%\n", K, avg_recall * 100.0f);
    printf("    Speedup vs flat: %.2fx\n", flat_mean / ivf_mean);

    EXPECT_GT(avg_recall, 0.7f) << "Recall too low";
}

// ============================================================================
// Recall sweep: Test different nprobe values
// ============================================================================
TEST_F(IVFBenchmark, RecallSweep_10Kx128) {
    constexpr uint64_t N = 10'000;
    constexpr uint32_t DIM = 128;
    constexpr uint32_t K = 10;
    constexpr uint32_t NLIST = 32;

    auto data = generate_clustered_vectors(N, DIM, NLIST, 42);
    auto queries = generate_vectors(10, DIM, 123);

    // Create index
    IVFParams ivf_params{.nlist = NLIST, .nprobe = NLIST, .kmeans_iters = 10};  // Start with full nprobe
    auto ivf_result = IndexIVFFlat::create(*ctx_, DIM, ivf_params);
    ASSERT_TRUE(ivf_result.has_value()) << ivf_result.error().message();
    auto ivf = std::move(*ivf_result);

    auto train_result = ivf.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = ivf.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    printf("\n  Recall sweep (10K x128, nlist=%u):\n", NLIST);
    printf("    %-8s %-10s %-12s\n", "nprobe", "recall@10", "median ms");

    for (uint32_t nprobe = 1; nprobe <= NLIST; nprobe *= 2) {
        ivf.set_nprobe(nprobe);

        // Compute recall
        float total_recall = 0.0f;
        for (int i = 0; i < 10; ++i) {
            std::vector<float> query(queries.begin() + i * DIM,
                                      queries.begin() + (i + 1) * DIM);
            auto gt = brute_force_search(data, query, N, DIM, K);
            auto result = ivf.search(query, K);
            ASSERT_TRUE(result.has_value());
            total_recall += compute_recall(gt, *result, K);
        }
        float avg_recall = total_recall / 10.0f;

        // Measure latency
        std::vector<double> times;
        for (int i = 0; i < 20; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto result = ivf.search({queries.data() + (i % 10) * DIM, DIM}, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double median = median_latency_ms(times);

        printf("    %-8u %-10.1f %-12.3f\n", nprobe, avg_recall * 100.0f, median);
    }
}

} // namespace
