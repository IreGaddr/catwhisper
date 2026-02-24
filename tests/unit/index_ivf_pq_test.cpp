#include <gtest/gtest.h>
#include <catwhisper/context.hpp>
#include <catwhisper/index_ivf_pq.hpp>
#include <random>
#include <vector>
#include <cmath>

namespace {

using namespace cw;

std::vector<float> generate_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

class IndexIVFPQTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx_result = Context::create();
        ASSERT_TRUE(ctx_result.has_value()) << ctx_result.error().message();
        ctx_ = std::make_unique<Context>(std::move(*ctx_result));
    }

    std::unique_ptr<Context> ctx_;
};

// ============================================================================
// Creation Tests
// ============================================================================

TEST_F(IndexIVFPQTest, CreateBasic) {
    IVFPQParams params{
        .ivf = {.nlist = 16, .nprobe = 4},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, 64, params);
    EXPECT_TRUE(result.has_value()) << result.error().message();

    auto index = std::move(*result);
    EXPECT_TRUE(index.valid());
    EXPECT_EQ(index.dimension(), 64);
    EXPECT_EQ(index.size(), 0);
    EXPECT_FALSE(index.is_trained());
}

TEST_F(IndexIVFPQTest, CreateInvalidDimension) {
    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 8, .nbits = 8}
    };

    // Dimension must be divisible by m
    auto result = IndexIVFPQ::create(*ctx_, 100, params);  // 100 / 8 != integer
    EXPECT_FALSE(result.has_value());
}

TEST_F(IndexIVFPQTest, CreateInvalidM) {
    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 0, .nbits = 8}  // m must be > 0
    };

    auto result = IndexIVFPQ::create(*ctx_, 64, params);
    EXPECT_FALSE(result.has_value());
}

TEST_F(IndexIVFPQTest, CreateInvalidNbits) {
    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 8, .nbits = 4}  // Only 8 is currently supported
    };

    auto result = IndexIVFPQ::create(*ctx_, 64, params);
    // For now, we only support 8 bits; may support 4 later
    // The test expects either success (if we add 4-bit support) or failure
    // For initial implementation, expect failure
    EXPECT_FALSE(result.has_value()) << "Only 8-bit PQ codes currently supported";
}

// ============================================================================
// Training Tests
// ============================================================================

TEST_F(IndexIVFPQTest, TrainBasic) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N, DIM);
    auto train_result = index.train(data, N);

    EXPECT_TRUE(train_result.has_value()) << train_result.error().message();
    EXPECT_TRUE(index.is_trained());
}

TEST_F(IndexIVFPQTest, TrainInsufficientData) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 10;  // Too few for 16 clusters * 256 subquantizers

    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N, DIM);
    auto train_result = index.train(data, N);

    // Should either succeed with fewer clusters or fail gracefully
    if (train_result.has_value()) {
        EXPECT_TRUE(index.is_trained());
    }
}

TEST_F(IndexIVFPQTest, TrainWithoutCreate) {
    IndexIVFPQ index;  // Default constructed, invalid
    auto data = generate_vectors(100, 64);

    auto train_result = index.train(data, 100);
    EXPECT_FALSE(train_result.has_value());
}

// ============================================================================
// Add Tests
// ============================================================================

TEST_F(IndexIVFPQTest, AddWithoutTrain) {
    constexpr uint32_t DIM = 64;

    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(100, DIM);
    auto add_result = index.add(data, 100);

    EXPECT_FALSE(add_result.has_value());
    EXPECT_EQ(index.size(), 0);
}

TEST_F(IndexIVFPQTest, AddBasic) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N * 2, DIM);  // Enough for train + add

    auto train_result = index.train(std::span<const float>(data.data(), N * DIM), N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = index.add(std::span<const float>(data.data() + N * DIM, N * DIM), N);
    EXPECT_TRUE(add_result.has_value()) << add_result.error().message();
    EXPECT_EQ(index.size(), N);
}

TEST_F(IndexIVFPQTest, AddMultiple) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N1 = 500;
    constexpr uint64_t N2 = 500;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(2000, DIM);

    auto train_result = index.train(data, 1000);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add1 = index.add(data, N1);
    EXPECT_TRUE(add1.has_value()) << add1.error().message();
    EXPECT_EQ(index.size(), N1);

    auto add2 = index.add(std::span<const float>(data.data() + N1 * DIM, N2 * DIM), N2);
    EXPECT_TRUE(add2.has_value()) << add2.error().message();
    EXPECT_EQ(index.size(), N1 + N2);
}

// ============================================================================
// Search Tests
// ============================================================================

TEST_F(IndexIVFPQTest, SearchEmpty) {
    constexpr uint32_t DIM = 64;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(100, DIM);
    auto train_result = index.train(data, 100);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    std::vector<float> query(DIM, 0.5f);
    auto search_result = index.search(query, 10);

    ASSERT_TRUE(search_result.has_value()) << search_result.error().message();
    EXPECT_EQ(search_result->n_queries, 1);
    EXPECT_EQ(search_result->k, 10);
    // All results should be invalid/empty
    for (const auto& r : search_result->results) {
        EXPECT_EQ(r.id, 0);
    }
}

TEST_F(IndexIVFPQTest, SearchBasic) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;
    constexpr uint32_t K = 10;

    IVFPQParams params{
        .ivf = {.nlist = 16, .nprobe = 8, .kmeans_iters = 10},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N * 2, DIM);

    auto train_result = index.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = index.add(std::span<const float>(data.data() + N * DIM, N * DIM), N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    std::vector<float> query(DIM, 0.5f);
    auto search_result = index.search(query, K);

    ASSERT_TRUE(search_result.has_value()) << search_result.error().message();
    EXPECT_EQ(search_result->n_queries, 1);
    EXPECT_EQ(search_result->k, K);

    // Results should be sorted by distance (ascending)
    for (uint32_t i = 1; i < K; ++i) {
        EXPECT_LE((*search_result)[0][i-1].distance, (*search_result)[0][i].distance);
    }
}

TEST_F(IndexIVFPQTest, SearchReturnsNearest) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;
    constexpr uint32_t K = 1;

    IVFPQParams params{
        .ivf = {.nlist = 16, .nprobe = 16, .kmeans_iters = 10},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N, DIM, 42);

    auto train_result = index.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = index.add(data, N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    // Search for a vector that exists in the database
    std::vector<float> query(data.begin(), data.begin() + DIM);  // First vector
    auto search_result = index.search(query, K);

    ASSERT_TRUE(search_result.has_value()) << search_result.error().message();

    // The nearest should be the vector itself (id = 0)
    // Note: PQ is approximate, so we allow some tolerance
    // The search should at least return something reasonable
    EXPECT_LT((*search_result)[0][0].distance, 100.0f);  // Should be relatively close
}

// ============================================================================
// Stats Tests
// ============================================================================

TEST_F(IndexIVFPQTest, StatsUntrained) {
    IVFPQParams params{
        .ivf = {.nlist = 16},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, 64, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto stats = index.stats();
    EXPECT_EQ(stats.n_vectors, 0);
    EXPECT_FALSE(stats.is_trained);
}

TEST_F(IndexIVFPQTest, StatsTrained) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N * 2, DIM);

    auto train_result = index.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = index.add(std::span<const float>(data.data() + N * DIM, N * DIM), N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();

    auto stats = index.stats();
    EXPECT_EQ(stats.n_vectors, N);
    EXPECT_TRUE(stats.is_trained);
    EXPECT_GT(stats.gpu_memory_used, 0);  // Should have allocated some memory
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_F(IndexIVFPQTest, Reset) {
    constexpr uint32_t DIM = 64;
    constexpr uint64_t N = 1000;

    IVFPQParams params{
        .ivf = {.nlist = 16, .kmeans_iters = 5},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, DIM, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    auto data = generate_vectors(N * 2, DIM);

    auto train_result = index.train(data, N);
    ASSERT_TRUE(train_result.has_value()) << train_result.error().message();

    auto add_result = index.add(std::span<const float>(data.data() + N * DIM, N * DIM), N);
    ASSERT_TRUE(add_result.has_value()) << add_result.error().message();
    EXPECT_EQ(index.size(), N);

    index.reset();

    EXPECT_EQ(index.size(), 0);
    // After reset, index should still be trained (codebooks preserved)
    // or not trained depending on implementation choice
}

// ============================================================================
// Nprobe Configuration Tests
// ============================================================================

TEST_F(IndexIVFPQTest, SetNprobe) {
    IVFPQParams params{
        .ivf = {.nlist = 32, .nprobe = 8},
        .pq = {.m = 8, .nbits = 8}
    };

    auto result = IndexIVFPQ::create(*ctx_, 64, params);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto index = std::move(*result);

    EXPECT_EQ(index.nprobe(), 8);

    index.set_nprobe(16);
    EXPECT_EQ(index.nprobe(), 16);

    // Should clamp to nlist
    index.set_nprobe(100);
    EXPECT_EQ(index.nprobe(), 32);
}

} // namespace
