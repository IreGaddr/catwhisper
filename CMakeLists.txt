cmake_minimum_required(VERSION 3.20)

project(catwhisper
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated vector similarity query library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CW_BUILD_TESTS "Build unit tests" ON)
option(CW_BUILD_EXAMPLES "Build example programs" ON)
option(CW_BUILD_SHARED "Build shared library" ON)
option(CW_USE_SYSTEM_VMA "Use system-provided VMA" OFF)
option(CW_ENABLE_VALIDATION "Enable Vulkan validation (debug only)" ON)
option(CW_ENABLE_NPU "Enable AMD XDNA NPU backend for batch similarity" OFF)

find_package(Vulkan REQUIRED)
find_package(OpenMP)

if(CW_USE_SYSTEM_VMA)
    find_package(VulkanMemoryAllocator REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        VMA
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.1.0
    )
    set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(VMA)
endif()

file(GLOB_RECURSE CW_SOURCES "src/*.cpp")
file(GLOB_RECURSE CW_HEADERS "include/*.hpp")

# NPU backend: conditionally include/exclude sources.
if(NOT CW_ENABLE_NPU)
    list(FILTER CW_SOURCES EXCLUDE REGEX "src/npu/.*\\.cpp$")
    list(FILTER CW_HEADERS EXCLUDE REGEX "npu_backend\\.hpp$|index_flat_npu\\.hpp$")
endif()

if(CW_BUILD_SHARED)
    add_library(catwhisper SHARED ${CW_SOURCES} ${CW_HEADERS})
else()
    add_library(catwhisper STATIC ${CW_SOURCES} ${CW_HEADERS})
endif()

add_library(catwhisper::catwhisper ALIAS catwhisper)

target_include_directories(catwhisper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(catwhisper
    PUBLIC
        Vulkan::Vulkan
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(catwhisper PRIVATE OpenMP::OpenMP_CXX)
endif()

# VMA linking (header-only with VMA_IMPLEMENTATION in buffer.cpp)
FetchContent_GetProperties(VMA)
if(VMA_SOURCE_DIR)
    target_include_directories(catwhisper PUBLIC ${VMA_SOURCE_DIR}/include)
endif()

if(CW_ENABLE_VALIDATION)
    target_compile_definitions(catwhisper
        PRIVATE
            $<$<CONFIG:Debug>:CW_DEBUG>
            $<$<CONFIG:Debug>:CW_VULKAN_VALIDATION>
    )
endif()

if(WIN32)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_MACOS_MVK)
elseif(UNIX)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_XCB_KHR)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # AMD Zen 4 (Ryzen 7 H 255) optimizations
    # -march=znver4 targets Zen 4 specifically for optimal AVX512/FP16 performance
    # -mtune=znver4 tunes for Zen 4 microarchitecture
    # -fno-plt avoids PLT overhead for better performance
    # -ffast-math enables aggressive floating-point optimizations
    target_compile_options(catwhisper PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Werror=return-type
        $<$<CONFIG:Release>:-march=znver4>
        $<$<CONFIG:Release>:-mtune=znver4>
        $<$<CONFIG:Release>:-ftree-vectorize>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fno-plt>
        $<$<CONFIG:Release>:-funroll-loops>
    )
    # Enable AVX512, F16C, VNNI, and other SIMD extensions
    target_compile_options(catwhisper PRIVATE
        $<$<CONFIG:Release>:-mavx512f>
        $<$<CONFIG:Release>:-mavx512dq>
        $<$<CONFIG:Release>:-mavx512vl>
        $<$<CONFIG:Release>:-mavx512bw>
        $<$<CONFIG:Release>:-mavx512bf16>
        $<$<CONFIG:Release>:-mavx512vnni>
        $<$<CONFIG:Release>:-mf16c>
        $<$<CONFIG:Release>:-mfma>
    )
elseif(MSVC)
    target_compile_options(catwhisper PRIVATE /W4 /wd4100 /fp:fast)
endif()

# NPU (AMD XDNA) support
if(CW_ENABLE_NPU)
    # Look for XDNA driver headers / Ryzen AI SDK.
    find_path(XDNA_INCLUDE_DIR
        NAMES "xrt/xrt_device.h" "amdxdna_accel.h"
        PATHS /usr/include /usr/local/include /opt/xilinx/xrt/include
    )
    find_library(XDNA_LIBRARY
        NAMES xrt_coreutil xrt_core
        PATHS /usr/lib /usr/local/lib /opt/xilinx/xrt/lib
    )

    if(XDNA_INCLUDE_DIR AND XDNA_LIBRARY)
        message(STATUS "AMD XDNA NPU support enabled (SDK: ${XDNA_INCLUDE_DIR})")
        target_compile_definitions(catwhisper PUBLIC CW_HAS_NPU=1)
        target_include_directories(catwhisper PRIVATE ${XDNA_INCLUDE_DIR})
        target_link_libraries(catwhisper PRIVATE ${XDNA_LIBRARY})
    else()
        message(STATUS "AMD XDNA NPU: SDK not found, building NPU backend with CPU reference kernels")
        message(STATUS "  Install Ryzen AI SDK or amdxdna driver for hardware acceleration")
        # Still compile NPU sources but without CW_HAS_NPU -- uses CPU reference path.
    endif()
endif()

add_subdirectory(shaders)

if(CW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
install(TARGETS catwhisper
    EXPORT catwhisper-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/catwhisper
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
