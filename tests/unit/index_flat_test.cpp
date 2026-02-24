#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>
#include <set>

class IndexFlatTest : public ::testing::Test {
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
    
    std::unique_ptr<cw::Context> ctx_;
};

TEST_F(IndexFlatTest, CreateDefault) {
    auto index = cw::IndexFlat::create(*ctx_, 128);
    
    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }
    
    EXPECT_TRUE(index->valid());
    EXPECT_EQ(index->dimension(), 128);
    EXPECT_EQ(index->size(), 0);
    EXPECT_TRUE(index->is_trained());
}

TEST_F(IndexFlatTest, AddVectors) {
    auto index = cw::IndexFlat::create(*ctx_, 64);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> data(100 * 64, 0.5f);
    
    auto result = index->add(data, 100);
    ASSERT_TRUE(result) << result.error().message();
    
    EXPECT_EQ(index->size(), 100);
}

TEST_F(IndexFlatTest, AddWithIds) {
    auto index = cw::IndexFlat::create(*ctx_, 32);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> data(50 * 32, 1.0f);
    std::vector<cw::VectorId> ids(50);
    for (uint64_t i = 0; i < 50; ++i) {
        ids[i] = 1000 + i;
    }
    
    auto result = index->add(data, 50, ids);
    ASSERT_TRUE(result) << result.error().message();
    
    EXPECT_EQ(index->size(), 50);
}

TEST_F(IndexFlatTest, SearchSingle) {
    auto index = cw::IndexFlat::create(*ctx_, 16);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> data(100 * 16);
    for (uint64_t i = 0; i < 100; ++i) {
        for (uint32_t j = 0; j < 16; ++j) {
            data[i * 16 + j] = static_cast<float>(i);
        }
    }
    
    auto add_result = index->add(data, 100);
    ASSERT_TRUE(add_result) << add_result.error().message();
    
    std::vector<float> query(16, 50.0f);
    
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    EXPECT_EQ(search_result->n_queries, 1);
    EXPECT_EQ(search_result->k, 5);
    
    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 5);
    
    EXPECT_EQ(results[0].id, 50);
    EXPECT_NEAR(results[0].distance, 0.0f, 0.001f);
}

TEST_F(IndexFlatTest, SearchBatch) {
    auto index = cw::IndexFlat::create(*ctx_, 8);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> data(100 * 8);
    for (uint64_t i = 0; i < 100; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            data[i * 8 + j] = static_cast<float>(i);
        }
    }
    
    auto add_result = index->add(data, 100);
    ASSERT_TRUE(add_result) << add_result.error().message();
    
    std::vector<float> queries(10 * 8);
    for (uint64_t q = 0; q < 10; ++q) {
        for (uint32_t j = 0; j < 8; ++j) {
            queries[q * 8 + j] = static_cast<float>(q * 10);
        }
    }
    
    auto search_result = index->search(queries, 10, 3);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    EXPECT_EQ(search_result->n_queries, 10);
    EXPECT_EQ(search_result->k, 3);
    
    for (uint64_t q = 0; q < 10; ++q) {
        auto results = (*search_result)[q];
        EXPECT_EQ(results.size(), 3);
    }
}

TEST_F(IndexFlatTest, SearchEmpty) {
    auto index = cw::IndexFlat::create(*ctx_, 16);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> query(16, 0.0f);
    
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    EXPECT_EQ(search_result->n_queries, 1);
}

TEST_F(IndexFlatTest, Reset) {
    auto index = cw::IndexFlat::create(*ctx_, 16);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> data(50 * 16, 0.0f);
    auto add_result = index->add(data, 50);
    ASSERT_TRUE(add_result);
    EXPECT_EQ(index->size(), 50);
    
    index->reset();
    EXPECT_EQ(index->size(), 0);
}

TEST_F(IndexFlatTest, InvalidDimension) {
    auto index = cw::IndexFlat::create(*ctx_, 32);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    std::vector<float> wrong_data(100 * 16, 0.0f);  // Wrong dimension
    
    auto result = index->add(wrong_data, 100);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code(), cw::ErrorCode::InvalidParameter);
}

TEST_F(IndexFlatTest, Stats) {
    auto index = cw::IndexFlat::create(*ctx_, 64);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }
    
    auto stats = index->stats();
    EXPECT_EQ(stats.n_vectors, 0);
    EXPECT_EQ(stats.dimension, 64);
    EXPECT_TRUE(stats.is_trained);
    
    std::vector<float> data(100 * 64, 0.0f);
    index->add(data, 100);
    
    stats = index->stats();
    EXPECT_EQ(stats.n_vectors, 100);
}

TEST_F(IndexFlatTest, InnerProductMetric) {
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;
    
    auto index = cw::IndexFlat::create(*ctx_, 8, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }
    
    EXPECT_TRUE(index->valid());
    
    // Add normalized vectors
    std::vector<float> data(5 * 8);
    // Vector 0: [1,0,0,0,0,0,0,0]
    data[0] = 1.0f;
    // Vector 1: [0,1,0,0,0,0,0,0]
    data[8 + 1] = 1.0f;
    // Vector 2: [0.707, 0.707, 0,0,0,0,0,0]
    data[16] = 0.707f;
    data[17] = 0.707f;
    // Vector 3: [-1,0,0,0,0,0,0,0]
    data[24] = -1.0f;
    // Vector 4: [0,0,0,0,0,0,0,1]
    data[32 + 7] = 1.0f;
    
    auto add_result = index->add(data, 5);
    ASSERT_TRUE(add_result) << add_result.error().message();
    
    // Query: [1,0,0,0,0,0,0,0] - should match vector 0 best (IP=1)
    std::vector<float> query(8, 0.0f);
    query[0] = 1.0f;
    
    auto search_result = index->search(query, 3);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 3);
    
    // Vector 0 should be first (IP = 1.0)
    EXPECT_EQ(results[0].id, 0);
    // For inner product, distance is negated IP, so -1.0
    EXPECT_NEAR(results[0].distance, -1.0f, 0.01f);
    
    // Vector 2 should be second (IP = 0.707)
    EXPECT_EQ(results[1].id, 2);
    EXPECT_NEAR(results[1].distance, -0.707f, 0.05f);
    
    // Vector 1 should be third (IP = 0)
    // Note: Vector 4 also has IP=0, so either could be third
    EXPECT_TRUE(results[2].id == 1 || results[2].id == 4);
}

TEST_F(IndexFlatTest, InnerProductBatchSearch) {
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;

    auto index = cw::IndexFlat::create(*ctx_, 4, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    // Add 3 vectors
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f, 0.0f,  // Vector 0
        0.0f, 1.0f, 0.0f, 0.0f,  // Vector 1
        0.0f, 0.0f, 1.0f, 0.0f,  // Vector 2
    };

    auto add_result = index->add(data, 3);
    ASSERT_TRUE(add_result);

    // Query 0 matches vector 0, Query 1 matches vector 1
    std::vector<float> queries = {
        1.0f, 0.0f, 0.0f, 0.0f,  // Query 0
        0.0f, 1.0f, 0.0f, 0.0f,  // Query 1
    };

    auto search_result = index->search(queries, 2, 2);
    ASSERT_TRUE(search_result);

    EXPECT_EQ((*search_result)[0][0].id, 0);  // Query 0 -> Vector 0
    EXPECT_EQ((*search_result)[1][0].id, 1);  // Query 1 -> Vector 1
}

// Exercises the multi-workgroup path: n_vectors > TOPK_CHUNK (512).
// Inserts 1000 vectors where vector i has all elements equal to float(i).
// Nearest to query [500, 500, ...] must be vector 500 (distance 0).
TEST_F(IndexFlatTest, SearchMultiWorkgroup) {
    auto index = cw::IndexFlat::create(*ctx_, 8);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    constexpr uint64_t N = 1000;
    std::vector<float> data(N * 8);
    for (uint64_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
            data[i * 8 + j] = static_cast<float>(i);
        }
    }
    auto add_result = index->add(data, N);
    ASSERT_TRUE(add_result) << add_result.error().message();

    std::vector<float> query(8, 500.0f);
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 5u);

    // The exact nearest neighbour must be vector 500 with L2 distance 0.
    EXPECT_EQ(results[0].id, 500u);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-3f);

    // Results must be sorted ascending by distance.
    for (uint32_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].distance, results[i].distance)
            << "Results not sorted at position " << i;
    }
}

// Verifies ascending sort order for the top-k results, including with
// negative distances (inner product metric).
TEST_F(IndexFlatTest, SearchSortOrderWithNegativeDistances) {
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;

    auto index = cw::IndexFlat::create(*ctx_, 4, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    // Vectors with known inner products against query [1,0,0,0].
    // IP values: 0.9, 0.5, 0.0, -0.3, 0.7
    // Distances (negated IP): -0.9, -0.5, 0.0, 0.3, -0.7
    // Expected rank (ascending dist): 0(-0.9), 4(-0.7), 1(-0.5), 2(0.0), 3(0.3)
    std::vector<float> data = {
        0.9f, 0.0f, 0.0f, 0.0f,   // Vector 0  IP=0.9  dist=-0.9
        0.5f, 0.0f, 0.0f, 0.0f,   // Vector 1  IP=0.5  dist=-0.5
        0.0f, 1.0f, 0.0f, 0.0f,   // Vector 2  IP=0.0  dist=0.0
       -0.3f, 0.0f, 0.0f, 0.0f,   // Vector 3  IP=-0.3 dist=0.3
        0.7f, 0.0f, 0.0f, 0.0f,   // Vector 4  IP=0.7  dist=-0.7
    };

    auto add_result = index->add(data, 5);
    ASSERT_TRUE(add_result) << add_result.error().message();

    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 5u);

    EXPECT_EQ(results[0].id, 0u);
    EXPECT_NEAR(results[0].distance, -0.9f, 0.02f);
    EXPECT_EQ(results[1].id, 4u);
    EXPECT_NEAR(results[1].distance, -0.7f, 0.02f);
    EXPECT_EQ(results[2].id, 1u);
    EXPECT_NEAR(results[2].distance, -0.5f, 0.02f);

    // Verify sort order across all 5 results.
    for (uint32_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].distance, results[i].distance)
            << "Results not sorted at position " << i;
    }
}

// Verifies that k > n_vectors is handled gracefully: search succeeds, returns
// k-length results, and the first n_vectors entries are correct nearest
// neighbours in sorted order.  The remaining entries are don't-care
// (implementation-defined default state).
TEST_F(IndexFlatTest, SearchKGreaterThanNVectors) {
    auto index = cw::IndexFlat::create(*ctx_, 4);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    // Vectors with distinct distances from query [1,0,0,0]:
    //   vec0  distance = 0    (exact match)
    //   vec1  distance = 2    (L2: (1-0)^2+(0-1)^2 = 2)
    //   vec2  distance = 2
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f, 0.0f,   // vec 0
        0.0f, 1.0f, 0.0f, 0.0f,   // vec 1
        0.0f, 0.0f, 1.0f, 0.0f,   // vec 2
    };
    auto add_result = index->add(data, 3);
    ASSERT_TRUE(add_result);

    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    // Ask for k=10 but only 3 vectors exist – must not crash.
    auto search_result = index->search(query, 10);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 10u);  // Always returns k entries.

    // The first entry must be the exact nearest (vec 0, distance 0).
    EXPECT_EQ(results[0].id, 0u);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-3f);

    // Entries 1 and 2 must be vec 1 and vec 2 (either order, both dist ≈ 2).
    std::set<cw::VectorId> second_tier = {results[1].id, results[2].id};
    EXPECT_TRUE(second_tier.count(1u) && second_tier.count(2u))
        << "Expected ids 1 and 2 in positions 1-2";
    EXPECT_NEAR(results[1].distance, 2.0f, 0.1f);
    EXPECT_NEAR(results[2].distance, 2.0f, 0.1f);
}

// Stress test with n_vectors that spans many workgroups.
// Verifies that the CPU merge step correctly selects the global nearest across
// multiple GPU workgroup outputs.
TEST_F(IndexFlatTest, SearchManyWorkgroups) {
    auto index = cw::IndexFlat::create(*ctx_, 4);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    // 2048 vectors: each vec[i] = [float(i), 0, 0, 0]
    // Query [2000, 0, 0, 0] → nearest is vec 2000, L2 dist = 0.
    constexpr uint64_t N = 2048;
    std::vector<float> data(N * 4, 0.0f);
    for (uint64_t i = 0; i < N; ++i) {
        data[i * 4] = static_cast<float>(i);
    }

    auto add_result = index->add(data, N);
    ASSERT_TRUE(add_result) << add_result.error().message();

    std::vector<float> query = {2000.0f, 0.0f, 0.0f, 0.0f};
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results[0].id, 2000u);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-3f);

    // Second nearest is vec 1999 or vec 2001.
    EXPECT_TRUE(results[1].id == 1999u || results[1].id == 2001u)
        << "Unexpected second nearest: " << results[1].id;

    for (uint32_t i = 1; i < 5; ++i) {
        EXPECT_LE(results[i - 1].distance, results[i].distance)
            << "Not sorted at position " << i;
    }
}
