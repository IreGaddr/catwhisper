#include <gtest/gtest.h>
#include <catwhisper/error.hpp>

TEST(ErrorTest, DefaultError) {
    cw::Error err;
    EXPECT_EQ(err.code(), cw::ErrorCode::Success);
    EXPECT_FALSE(err);
}

TEST(ErrorTest, ErrorWithMessage) {
    cw::Error err(cw::ErrorCode::BufferCreationFailed, "Test message");
    
    EXPECT_EQ(err.code(), cw::ErrorCode::BufferCreationFailed);
    EXPECT_EQ(err.message(), "Test message");
    EXPECT_TRUE(err);
}

TEST(ErrorTest, ErrorComparison) {
    cw::Error e1(cw::ErrorCode::BufferCreationFailed);
    cw::Error e2(cw::ErrorCode::BufferCreationFailed);
    cw::Error e3(cw::ErrorCode::AllocationFailed);
    
    EXPECT_EQ(e1, e2);
    EXPECT_NE(e1, e3);
}

TEST(ErrorTest, MakeUnexpected) {
    auto unexpected = cw::make_unexpected(cw::ErrorCode::InvalidParameter, "Bad param");
    
    EXPECT_EQ(unexpected.error().code(), cw::ErrorCode::InvalidParameter);
    EXPECT_EQ(unexpected.error().message(), "Bad param");
}

TEST(ErrorTest, ExpectedSuccess) {
    cw::Expected<int> result = 42;
    
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST(ErrorTest, ExpectedError) {
    cw::Expected<int> result = cw::make_unexpected(cw::ErrorCode::OperationFailed, "Failed");
    
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), cw::ErrorCode::OperationFailed);
}
