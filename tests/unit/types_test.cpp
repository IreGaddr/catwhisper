#include <gtest/gtest.h>
#include <catwhisper/types.hpp>

TEST(TypesTest, SearchResultDefault) {
    cw::SearchResult result;
    EXPECT_EQ(result.id, 0);
    EXPECT_EQ(result.distance, 0.0f);
}

TEST(TypesTest, SearchResultsBasic) {
    cw::SearchResults results(10, 5);
    
    EXPECT_EQ(results.n_queries, 10);
    EXPECT_EQ(results.k, 5);
    EXPECT_EQ(results.results.size(), 50);
}

TEST(TypesTest, SearchResultsAccess) {
    cw::SearchResults results(3, 2);
    
    results.results[0] = {1, 0.1f};
    results.results[1] = {2, 0.2f};
    results.results[2] = {3, 0.3f};
    results.results[3] = {4, 0.4f};
    results.results[4] = {5, 0.5f};
    results.results[5] = {6, 0.6f};
    
    auto q0 = results[0];
    EXPECT_EQ(q0.size(), 2);
    EXPECT_EQ(q0[0].id, 1);
    EXPECT_EQ(q0[1].id, 2);
    
    auto q2 = results[2];
    EXPECT_EQ(q2.size(), 2);
    EXPECT_EQ(q2[0].id, 5);
    EXPECT_EQ(q2[1].id, 6);
}

TEST(TypesTest, IndexStatsDefault) {
    cw::IndexStats stats;
    EXPECT_EQ(stats.n_vectors, 0);
    EXPECT_EQ(stats.dimension, 0);
    EXPECT_EQ(stats.memory_used, 0);
    EXPECT_EQ(stats.gpu_memory_used, 0);
    EXPECT_FALSE(stats.is_trained);
}

TEST(TypesTest, DeviceInfoDefault) {
    cw::DeviceInfo info;
    EXPECT_EQ(info.device_id, 0);
    EXPECT_TRUE(info.name.empty());
    EXPECT_EQ(info.total_memory, 0);
    EXPECT_FALSE(info.supports_fp16);
    EXPECT_FALSE(info.supports_int8);
}
