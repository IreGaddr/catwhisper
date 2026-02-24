cmake_minimum_required(VERSION 3.20)

project(catwhisper
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated vector similarity search library"
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

find_package(Vulkan REQUIRED)

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
    target_compile_options(catwhisper PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Werror=return-type
    )
elseif(MSVC)
    target_compile_options(catwhisper PRIVATE /W4 /wd4100)
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
