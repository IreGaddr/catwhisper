#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>

class ContextTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ContextTest, ListDevices) {
    auto devices = cw::Context::list_devices();
    
    if (!devices) {
        GTEST_SKIP() << "No Vulkan devices available: " << devices.error().message();
    }
    
    EXPECT_FALSE(devices->empty()) << "Should have at least one device";
    
    for (const auto& dev : *devices) {
        EXPECT_FALSE(dev.name.empty()) << "Device name should not be empty";
        EXPECT_GT(dev.total_memory, 0) << "Device should have memory";
    }
}

TEST_F(ContextTest, CreateDefault) {
    auto ctx = cw::Context::create();
    
    if (!ctx) {
        GTEST_SKIP() << "Failed to create context: " << ctx.error().message();
    }
    
    EXPECT_TRUE(ctx->valid());
    EXPECT_FALSE(ctx->device_info().name.empty());
    EXPECT_GT(ctx->total_gpu_memory(), 0);
}

TEST_F(ContextTest, CreateWithOptions) {
    cw::ContextOptions opts;
    opts.enable_validation = false;
    
    auto ctx = cw::Context::create(opts);
    
    if (!ctx) {
        GTEST_SKIP() << "Failed to create context: " << ctx.error().message();
    }
    
    EXPECT_TRUE(ctx->valid());
}

TEST_F(ContextTest, MoveConstructor) {
    auto ctx1 = cw::Context::create();
    if (!ctx1) {
        GTEST_SKIP() << "Failed to create context";
    }
    
    std::string name = ctx1->device_info().name;
    
    cw::Context ctx2(std::move(*ctx1));
    EXPECT_FALSE(ctx1->valid());
    EXPECT_TRUE(ctx2.valid());
    EXPECT_EQ(ctx2.device_info().name, name);
}

TEST_F(ContextTest, Synchronize) {
    auto ctx = cw::Context::create();
    if (!ctx) {
        GTEST_SKIP() << "Failed to create context";
    }
    
    EXPECT_NO_THROW(ctx->synchronize());
}
