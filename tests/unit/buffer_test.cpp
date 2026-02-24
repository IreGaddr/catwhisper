#include <gtest/gtest.h>
#include <catwhisper/catwhisper.hpp>

class BufferTest : public ::testing::Test {
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

TEST_F(BufferTest, CreateDeviceLocal) {
    cw::BufferDesc desc = {
        .size = 1024 * 1024,  // 1MB
        .usage = cw::BufferUsage::Storage,
        .memory_type = cw::MemoryType::DeviceLocal
    };
    
    auto buffer = cw::Buffer::create(*ctx_, desc);
    ASSERT_TRUE(buffer) << buffer.error().message();
    
    EXPECT_TRUE(buffer->valid());
    EXPECT_EQ(buffer->size(), 1024 * 1024);
}

TEST_F(BufferTest, CreateHostVisible) {
    cw::BufferDesc desc = {
        .size = 4096,
        .usage = cw::BufferUsage::Storage | cw::BufferUsage::TransferSrc,
        .memory_type = cw::MemoryType::HostVisible,
        .map_on_create = true
    };
    
    auto buffer = cw::Buffer::create(*ctx_, desc);
    ASSERT_TRUE(buffer) << buffer.error().message();
    
    EXPECT_TRUE(buffer->valid());
    EXPECT_NE(buffer->mapped(), nullptr);
}

TEST_F(BufferTest, UploadDownload) {
    cw::BufferDesc desc = {
        .size = 1024,
        .usage = cw::BufferUsage::Storage | cw::BufferUsage::TransferDst | cw::BufferUsage::TransferSrc,
        .memory_type = cw::MemoryType::HostVisible,
        .map_on_create = true
    };
    
    auto buffer = cw::Buffer::create(*ctx_, desc);
    ASSERT_TRUE(buffer) << buffer.error().message();
    
    std::vector<uint8_t> upload_data(1024);
    for (size_t i = 0; i < 1024; ++i) {
        upload_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    auto upload_result = buffer->upload(upload_data);
    ASSERT_TRUE(upload_result) << upload_result.error().message();
    
    std::vector<uint8_t> download_data(1024);
    auto download_result = buffer->download(download_data);
    ASSERT_TRUE(download_result) << download_result.error().message();
    
    EXPECT_EQ(upload_data, download_data);
}

TEST_F(BufferTest, TypedUpload) {
    cw::BufferDesc desc = {
        .size = 100 * sizeof(float),
        .usage = cw::BufferUsage::Storage,
        .memory_type = cw::MemoryType::HostVisible,
        .map_on_create = true
    };
    
    auto buffer = cw::Buffer::create(*ctx_, desc);
    ASSERT_TRUE(buffer);
    
    std::vector<float> data(100, 3.14f);
    auto result = buffer->upload_typed(std::span<const float>(data));
    ASSERT_TRUE(result) << result.error().message();
    
    std::vector<float> readback(100);
    auto download_result = buffer->download_typed(std::span<float>(readback));
    ASSERT_TRUE(download_result);
    
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(readback[i], 3.14f);
    }
}

TEST_F(BufferTest, MoveConstructor) {
    cw::BufferDesc desc = {
        .size = 1024,
        .usage = cw::BufferUsage::Storage,
        .memory_type = cw::MemoryType::DeviceLocal
    };
    
    auto buffer1 = cw::Buffer::create(*ctx_, desc);
    ASSERT_TRUE(buffer1);
    
    uint64_t size = buffer1->size();
    void* vk_buf = buffer1->vulkan_buffer();
    
    cw::Buffer buffer2(std::move(*buffer1));
    
    EXPECT_FALSE(buffer1->valid());
    EXPECT_TRUE(buffer2.valid());
    EXPECT_EQ(buffer2.size(), size);
    EXPECT_EQ(buffer2.vulkan_buffer(), vk_buf);
}
