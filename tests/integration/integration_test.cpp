#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx = cw::Context::create();
        if (!ctx) {
            GTEST_SKIP() << "Failed to create context: " << ctx.error().message();
        }
        ctx_ = std::make_unique<cw::Context>(std::move(*ctx));
    }
    
    void TearDown() override {
        ctx_.reset();
    }
    
    std::unique_ptr<cw::Context> ctx_;
};

TEST_F(IntegrationTest, EndToEndBasic) {
    const uint32_t dim = 64;
    const uint64_t n = 1000;
    
    auto index = cw::IndexFlat::create(*ctx_, dim);
    ASSERT_TRUE(index) << index.error().message();
    
    std::vector<float> data(n * dim);
    for (uint64_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<float>(i + j);
        }
    }
    
    auto add_result = index->add(data, n);
    ASSERT_TRUE(add_result) << add_result.error().message();
    
    EXPECT_EQ(index->size(), n);
    
    std::vector<float> query(data.begin(), data.begin() + dim);
    
    auto search_result = index->search(query, 10);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    auto results = (*search_result)[0];
    EXPECT_EQ(results[0].id, 0);
    EXPECT_NEAR(results[0].distance, 0.0f, 0.01f);
}

TEST_F(IntegrationTest, LargeDataset) {
    const uint32_t dim = 128;
    const uint64_t n = 50000;
    
    auto index = cw::IndexFlat::create(*ctx_, dim);
    ASSERT_TRUE(index) << index.error().message();
    
    std::vector<float> data(n * dim);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : data) {
        v = dist(rng);
    }
    
    auto add_result = index->add(data, n);
    ASSERT_TRUE(add_result) << add_result.error().message();
    
    EXPECT_EQ(index->size(), n);
    
    std::vector<float> query(data.begin(), data.begin() + dim);
    
    auto search_result = index->search(query, 5);
    ASSERT_TRUE(search_result) << search_result.error().message();
    
    EXPECT_EQ(search_result->k, 5);
}

TEST_F(IntegrationTest, MultipleSearches) {
    const uint32_t dim = 32;
    const uint64_t n = 1000;
    const int n_queries = 50;
    
    auto index = cw::IndexFlat::create(*ctx_, dim);
    ASSERT_TRUE(index);
    
    std::vector<float> data(n * dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) {
        v = dist(rng);
    }
    
    index->add(data, n);
    
    for (int q = 0; q < n_queries; ++q) {
        std::vector<float> query(dim);
        for (auto& v : query) {
            v = dist(rng);
        }
        
        auto result = index->search(query, 10);
        ASSERT_TRUE(result) << "Query " << q << " failed";
        EXPECT_EQ(result->k, 10);
    }
}

TEST_F(IntegrationTest, IncrementalAdd) {
    const uint32_t dim = 16;
    
    auto index = cw::IndexFlat::create(*ctx_, dim);
    ASSERT_TRUE(index);
    
    std::vector<float> batch1(100 * dim, 1.0f);
    std::vector<float> batch2(100 * dim, 2.0f);
    std::vector<float> batch3(100 * dim, 3.0f);
    
    index->add(batch1, 100);
    EXPECT_EQ(index->size(), 100);
    
    index->add(batch2, 100);
    EXPECT_EQ(index->size(), 200);
    
    index->add(batch3, 100);
    EXPECT_EQ(index->size(), 300);
    
    std::vector<float> query(dim, 2.0f);
    auto result = index->search(query, 5);
    ASSERT_TRUE(result);
    
    EXPECT_EQ((*result)[0][0].id, 100);
}
