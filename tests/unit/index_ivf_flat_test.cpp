#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>
#include <cmath>
#include <random>
#include <algorithm>

class IndexIVFFlatTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx = cw::Context::create();
        if (!ctx) {
            GTEST_SKIP() << "Failed to create context";
        }
        ctx_ = std::make_unique<cw::Context>(std::move(*ctx));
    }

    void TearDown() override {
        ctx_.reset();
    }

    std::vector<float> generate_random_data(uint64_t n_vectors, uint32_t dim, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(n_vectors * dim);
        for (auto& v : data) v = dist(rng);
        return data;
    }

    std::vector<float> generate_clustered_data(
        uint64_t n_vectors, uint32_t dim, uint32_t n_clusters, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> cluster_center_dist(0.0f, 10.0f);
        std::normal_distribution<float> noise_dist(0.0f, 1.0f);

        std::vector<float> data(n_vectors * dim);
        std::vector<std::vector<float>> centers(n_clusters);
        for (auto& center : centers) {
            center.resize(dim);
            for (uint32_t d = 0; d < dim; ++d) {
                center[d] = cluster_center_dist(rng);
            }
        }

        std::uniform_int_distribution<uint32_t> cluster_dist(0, n_clusters - 1);
        for (uint64_t i = 0; i < n_vectors; ++i) {
            uint32_t cluster = cluster_dist(rng);
            for (uint32_t d = 0; d < dim; ++d) {
                data[i * dim + d] = centers[cluster][d] + noise_dist(rng);
            }
        }
        return data;
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

        uint32_t actual_k = std::min(k, static_cast<uint32_t>(n_vectors));
        std::partial_sort(results.begin(), results.begin() + actual_k, results.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        results.resize(actual_k);
        return results;
    }

    float compute_recall(
        const std::vector<std::pair<float, uint64_t>>& ground_truth,
        const cw::SearchResults& results, uint32_t k) {

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

    std::unique_ptr<cw::Context> ctx_;
};

// Construction tests
TEST_F(IndexIVFFlatTest, CreateDefault) {
    cw::IVFParams params{.nlist = 16, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 128, params);

    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }

    EXPECT_TRUE(index->valid());
    EXPECT_EQ(index->dimension(), 128);
    EXPECT_EQ(index->size(), 0);
    EXPECT_FALSE(index->is_trained());
}

TEST_F(IndexIVFFlatTest, CreateWithIndexOptions) {
    cw::IVFParams params{.nlist = 32, .nprobe = 8};
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;

    auto index = cw::IndexIVFFlat::create(*ctx_, 64, params, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }

    EXPECT_TRUE(index->valid());
    EXPECT_EQ(index->dimension(), 64);
}

// Training tests
TEST_F(IndexIVFFlatTest, TrainBasic) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(1000, 16, 8);
    auto train_result = index->train(train_data, 1000);
    ASSERT_TRUE(train_result) << train_result.error().message();
    EXPECT_TRUE(index->is_trained());
}

TEST_F(IndexIVFFlatTest, TrainTwiceRetrains) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    auto train_data2 = generate_clustered_data(500, 16, 8, 123);
    auto train2 = index->train(train_data2, 500);
    ASSERT_TRUE(train2);
}

// Add vectors tests
TEST_F(IndexIVFFlatTest, AddWithoutTraining) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    std::vector<float> data(100 * 16, 0.5f);
    auto add_result = index->add(data, 100);
    EXPECT_FALSE(add_result);
}

TEST_F(IndexIVFFlatTest, AddAfterTraining) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    std::vector<float> add_data(100 * 16, 0.5f);
    auto add_result = index->add(add_data, 100);
    ASSERT_TRUE(add_result) << add_result.error().message();
    EXPECT_EQ(index->size(), 100);
}

TEST_F(IndexIVFFlatTest, AddWithIds) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    std::vector<float> data(50 * 16, 1.0f);
    std::vector<cw::VectorId> ids(50);
    for (uint64_t i = 0; i < 50; ++i) {
        ids[i] = 1000 + i;
    }

    auto add_result = index->add(data, 50, ids);
    ASSERT_TRUE(add_result) << add_result.error().message();
    EXPECT_EQ(index->size(), 50);
}

// Search tests
TEST_F(IndexIVFFlatTest, SearchAfterTrainAndAdd) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(1000, 16, 8);
    ASSERT_TRUE(index->train(train_data, 1000));

    auto add_data = generate_clustered_data(500, 16, 8, 999);
    ASSERT_TRUE(index->add(add_data, 500));

    std::vector<float> query(16, 0.5f);
    auto search_result = index->search(query, 10);
    ASSERT_TRUE(search_result) << search_result.error().message();

    EXPECT_EQ(search_result->n_queries, 1);
    EXPECT_EQ(search_result->k, 10);
}

TEST_F(IndexIVFFlatTest, SearchEmptyIndex) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    std::vector<float> query(16, 0.0f);
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();
    EXPECT_EQ(search_result->n_queries, 1);
}

TEST_F(IndexIVFFlatTest, SearchBatch) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(1000, 16, 8);
    ASSERT_TRUE(index->train(train_data, 1000));

    auto add_data = generate_clustered_data(200, 16, 8, 777);
    ASSERT_TRUE(index->add(add_data, 200));

    std::vector<float> queries(5 * 16, 0.5f);
    auto search_result = index->search(queries, 5, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();

    EXPECT_EQ(search_result->n_queries, 5);
    EXPECT_EQ(search_result->k, 5);
}

// Recall test - basic correctness check
TEST_F(IndexIVFFlatTest, SearchReturnsCorrectResults) {
    cw::IVFParams params{.nlist = 8, .nprobe = 8};  // nprobe = nlist = full scan
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    auto add_data = generate_clustered_data(100, 16, 8, 111);
    ASSERT_TRUE(index->add(add_data, 100));

    // Query with nprobe=nlist should give same results as brute force
    std::vector<float> query(16, 0.5f);
    auto gt = brute_force_search(add_data, query, 100, 16, 5);
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result);

    float recall = compute_recall(gt, *search_result, 5);
    // With nprobe = nlist (full scan), recall should be 100%
    EXPECT_FLOAT_EQ(recall, 1.0f);
}

// Stats test
TEST_F(IndexIVFFlatTest, Stats) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 32, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto stats = index->stats();
    EXPECT_EQ(stats.n_vectors, 0);
    EXPECT_EQ(stats.dimension, 32);
    EXPECT_FALSE(stats.is_trained);

    auto train_data = generate_clustered_data(500, 32, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    stats = index->stats();
    EXPECT_TRUE(stats.is_trained);

    std::vector<float> data(100 * 32, 0.0f);
    ASSERT_TRUE(index->add(data, 100));

    stats = index->stats();
    EXPECT_EQ(stats.n_vectors, 100);
}

// Reset test
TEST_F(IndexIVFFlatTest, Reset) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 16, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 16, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    std::vector<float> data(50 * 16, 0.0f);
    ASSERT_TRUE(index->add(data, 50));
    EXPECT_EQ(index->size(), 50);

    index->reset();
    EXPECT_EQ(index->size(), 0);
}

// Invalid dimension test
TEST_F(IndexIVFFlatTest, InvalidDimension) {
    cw::IVFParams params{.nlist = 8, .nprobe = 4};
    auto index = cw::IndexIVFFlat::create(*ctx_, 32, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    auto train_data = generate_clustered_data(500, 32, 8);
    ASSERT_TRUE(index->train(train_data, 500));

    std::vector<float> wrong_data(100 * 16, 0.0f);

    auto result = index->add(wrong_data, 100);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code(), cw::ErrorCode::InvalidParameter);
}

// Inner product metric test
TEST_F(IndexIVFFlatTest, InnerProductMetric) {
    cw::IVFParams params{.nlist = 4, .nprobe = 4};
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;

    auto index = cw::IndexIVFFlat::create(*ctx_, 8, params, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }

    auto train_data = generate_random_data(200, 8, 42);
    auto train_result = index->train(train_data, 200);
    ASSERT_TRUE(train_result) << train_result.error().message();

    // Add specific vectors with known IP relationships
    std::vector<float> add_data(5 * 8, 0.0f);
    add_data[0] = 1.0f;  // [1,0,0,...]
    add_data[8 + 1] = 1.0f;  // [0,1,0,...]
    add_data[16] = 0.707f;  // [0.707, 0.707, 0,...]
    add_data[17] = 0.707f;
    add_data[24] = -1.0f;  // [-1,0,0,...]

    auto add_result = index->add(add_data, 5);
    ASSERT_TRUE(add_result);

    // Query [1,0,0,...] - vector 0 should have highest IP (1.0)
    std::vector<float> query(8, 0.0f);
    query[0] = 1.0f;

    auto search_result = index->search(query, 3);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_GT(results.size(), 0u);
    EXPECT_EQ(results[0].id, 0u);  // First result should be vector 0
}
