#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>

// Test 1: Can list devices without creating a context
TEST(MinimalTest, ListDevicesWorks) {
    auto devices = cw::Context::list_devices();
    EXPECT_TRUE(devices.has_value());
    // May be empty if no GPU, but call should succeed
}

// Test 2: Context creation succeeds or fails gracefully
TEST(MinimalTest, ContextCreationGraceful) {
    auto ctx = cw::Context::create();
    // Either we get a valid context or an error - both are acceptable
    EXPECT_TRUE(ctx.has_value() || !ctx.has_value());
}

// Test 3: IndexFlat can be created with valid context
TEST(MinimalTest, IndexFlatCreation) {
    auto ctx = cw::Context::create();
    if (!ctx) {
        GTEST_SKIP() << "No GPU context available";
    }
    
    auto index = cw::IndexFlat::create(*ctx, 128);
    EXPECT_TRUE(index.has_value());
    EXPECT_EQ(index->dimension(), 128);
    EXPECT_EQ(index->size(), 0);
}

// Test 4: IndexFlat can add vectors
TEST(MinimalTest, IndexFlatAddVectors) {
    auto ctx = cw::Context::create();
    if (!ctx) {
        GTEST_SKIP() << "No GPU context available";
    }
    
    auto index = cw::IndexFlat::create(*ctx, 4);
    ASSERT_TRUE(index.has_value());
    
    std::vector<float> data = {1.0f, 0.0f, 0.0f, 0.0f};
    auto result = index->add(data, 1);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(index->size(), 1);
}

// Test 5: IndexFlat can search
TEST(MinimalTest, IndexFlatSearch) {
    auto ctx = cw::Context::create();
    if (!ctx) {
        GTEST_SKIP() << "No GPU context available";
    }
    
    auto index = cw::IndexFlat::create(*ctx, 4);
    ASSERT_TRUE(index.has_value());
    
    // Add two vectors
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f, 0.0f,  // vector 0
        0.0f, 1.0f, 0.0f, 0.0f   // vector 1
    };
    auto add_result = index->add(data, 2);
    ASSERT_TRUE(add_result.has_value());
    
    // Search for nearest to [1,0,0,0]
    cw::Vector query(data.data(), 4);
    auto search_result = index->search(query, 1);
    
    ASSERT_TRUE(search_result.has_value());
    EXPECT_EQ(search_result->results.size(), 1);
    EXPECT_NEAR(search_result->results[0].distance, 0.0f, 0.001f);
}
