#ifndef CATWHISPER_ERROR_HPP
#define CATWHISPER_ERROR_HPP

#include <expected>
#include <string>

namespace cw {

enum class ErrorCode : int {
    Success = 0,

    VulkanInitFailed = 100,
    NoComputeCapableDevice = 101,
    DeviceCreationFailed = 102,
    InstanceCreationFailed = 103,

    OutOfGPUMemory = 200,
    OutOfHostMemory = 201,
    BufferCreationFailed = 202,
    AllocationFailed = 203,

    IndexNotTrained = 300,
    InvalidDimension = 301,
    InvalidParameter = 302,
    IndexFull = 303,

    FileNotFound = 400,
    InvalidFileFormat = 401,
    WriteFailed = 402,
    ReadFailed = 403,

    OperationFailed = 500,
    Timeout = 501,
    DeviceLost = 502,
    ShaderCompilationFailed = 503,
    PipelineCreationFailed = 504
};

class Error {
public:
    Error() : code_(ErrorCode::Success), message_() {}

    Error(ErrorCode code, std::string message = "")
        : code_(code), message_(std::move(message)) {}

    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }

    explicit operator bool() const { return code_ != ErrorCode::Success; }

    friend bool operator==(const Error& lhs, const Error& rhs) {
        return lhs.code_ == rhs.code_;
    }

    friend bool operator!=(const Error& lhs, const Error& rhs) {
        return lhs.code_ != rhs.code_;
    }

private:
    ErrorCode code_;
    std::string message_;
};

template<typename T>
using Expected = std::expected<T, Error>;

using Unexpected = std::unexpected<Error>;

inline Unexpected make_unexpected(ErrorCode code, std::string message = "") {
    return Unexpected(Error(code, std::move(message)));
}

} // namespace cw

#endif // CATWHISPER_ERROR_HPP
