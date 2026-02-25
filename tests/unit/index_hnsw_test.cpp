#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>
#include <set>
#include <cmath>

class IndexHNSWTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(IndexHNSWTest, CreateDefault) {
    auto index = cw::IndexHNSW::create(128);

    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }

    EXPECT_TRUE(index->valid());
    EXPECT_EQ(index->dimension(), 128);
    EXPECT_EQ(index->size(), 0);
    EXPECT_TRUE(index->is_trained());
}

TEST_F(IndexHNSWTest, CreateWithParams) {
    cw::HNSWParams params;
    params.M = 32;
    params.ef_construction = 256;

    auto index = cw::IndexHNSW::create(64, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    EXPECT_TRUE(index->valid());
    EXPECT_EQ(index->dimension(), 64);
}

TEST_F(IndexHNSWTest, AddVectors) {
    auto index = cw::IndexHNSW::create(64);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    std::vector<float> data(100 * 64, 0.5f);

    auto result = index->add(data, 100);
    ASSERT_TRUE(result) << result.error().message();

    EXPECT_EQ(index->size(), 100);
}

TEST_F(IndexHNSWTest, AddWithIds) {
    auto index = cw::IndexHNSW::create(32);
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

TEST_F(IndexHNSWTest, SearchSingle) {
    auto index = cw::IndexHNSW::create(16);
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

TEST_F(IndexHNSWTest, SearchBatch) {
    auto index = cw::IndexHNSW::create(8);
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

TEST_F(IndexHNSWTest, SearchEmpty) {
    auto index = cw::IndexHNSW::create(16);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    std::vector<float> query(16, 0.0f);

    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();

    EXPECT_EQ(search_result->n_queries, 1);
}

TEST_F(IndexHNSWTest, Reset) {
    auto index = cw::IndexHNSW::create(16);
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

TEST_F(IndexHNSWTest, InvalidDimension) {
    auto index = cw::IndexHNSW::create(32);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    std::vector<float> wrong_data(100 * 16, 0.0f);

    auto result = index->add(wrong_data, 100);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code(), cw::ErrorCode::InvalidParameter);
}

TEST_F(IndexHNSWTest, Stats) {
    auto index = cw::IndexHNSW::create(64);
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

TEST_F(IndexHNSWTest, EfSearch) {
    auto index = cw::IndexHNSW::create(16);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    EXPECT_EQ(index->ef_search(), 50);

    index->set_ef_search(100);
    EXPECT_EQ(index->ef_search(), 100);
}

TEST_F(IndexHNSWTest, InnerProductMetric) {
    cw::IndexOptions options;
    options.metric = cw::Metric::IP;

    auto index = cw::IndexHNSW::create(8, {}, options);
    if (!index) {
        GTEST_SKIP() << "Failed to create index: " << index.error().message();
    }

    EXPECT_TRUE(index->valid());

    std::vector<float> data(5 * 8);
    data[0] = 1.0f;
    data[8 + 1] = 1.0f;
    data[16] = 0.707f;
    data[17] = 0.707f;
    data[24] = -1.0f;
    data[32 + 7] = 1.0f;

    auto add_result = index->add(data, 5);
    ASSERT_TRUE(add_result) << add_result.error().message();

    std::vector<float> query(8, 0.0f);
    query[0] = 1.0f;

    auto search_result = index->search(query, 3);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 3);

    EXPECT_EQ(results[0].id, 0);
    EXPECT_NEAR(results[0].distance, -1.0f, 0.05f);
}

TEST_F(IndexHNSWTest, RecallTest) {
    cw::HNSWParams params;
    params.M = 16;
    params.ef_construction = 200;

    auto index = cw::IndexHNSW::create(32, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    constexpr uint64_t N = 1000;
    std::vector<float> data(N * 32);
    for (uint64_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < 32; ++j) {
            data[i * 32 + j] = static_cast<float>(i) + static_cast<float>(j) * 0.01f;
        }
    }

    auto add_result = index->add(data, N);
    ASSERT_TRUE(add_result) << add_result.error().message();

    index->set_ef_search(100);

    std::vector<float> query(32, 500.0f);
    auto search_result = index->search(query, 10);
    ASSERT_TRUE(search_result) << search_result.error().message();

    auto results = (*search_result)[0];
    EXPECT_EQ(results.size(), 10);

    EXPECT_NEAR(results[0].distance, 0.0f, 50.0f);
}

TEST_F(IndexHNSWTest, LargeIndex) {
    cw::HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;

    auto index = cw::IndexHNSW::create(64, params);
    if (!index) {
        GTEST_SKIP() << "Failed to create index";
    }

    constexpr uint64_t N = 10000;
    std::vector<float> data(N * 64);
    for (uint64_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < 64; ++j) {
            data[i * 64 + j] = static_cast<float>(i) * 0.01f + static_cast<float>(j) * 0.001f;
        }
    }

    auto add_result = index->add(data, N);
    ASSERT_TRUE(add_result) << add_result.error().message();

    EXPECT_EQ(index->size(), N);

    index->set_ef_search(64);
    std::vector<float> query(64, 50.0f);
    auto search_result = index->search(query, 10);
    ASSERT_TRUE(search_result) << search_result.error().message();
}

TEST_F(IndexHNSWTest, SerializationRoundTrip) {
    cw::HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;

    auto index1 = cw::IndexHNSW::create(32, params);
    if (!index1) {
        GTEST_SKIP() << "Failed to create index";
    }

    std::vector<float> data(100 * 32);
    for (uint64_t i = 0; i < 100; ++i) {
        for (uint32_t j = 0; j < 32; ++j) {
            data[i * 32 + j] = static_cast<float>(i);
        }
    }

    auto add_result = index1->add(data, 100);
    ASSERT_TRUE(add_result) << add_result.error().message();

    auto save_result = index1->save("/tmp/test_hnsw_index.cw");
    ASSERT_TRUE(save_result) << save_result.error().message();

    auto index2 = cw::IndexHNSW::create(32);
    ASSERT_TRUE(index2);

    auto load_result = index2->load("/tmp/test_hnsw_index.cw");
    ASSERT_TRUE(load_result) << load_result.error().message();

    EXPECT_EQ(index2->size(), 100);
    EXPECT_EQ(index2->dimension(), 32);

    std::vector<float> query(32, 50.0f);
    auto result1 = index1->search(query, 5);
    auto result2 = index2->search(query, 5);

    ASSERT_TRUE(result1);
    ASSERT_TRUE(result2);

    auto r1 = (*result1)[0];
    auto r2 = (*result2)[0];

    EXPECT_EQ(r1[0].id, r2[0].id);
}
