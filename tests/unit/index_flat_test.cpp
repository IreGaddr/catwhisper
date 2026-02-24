#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>

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
