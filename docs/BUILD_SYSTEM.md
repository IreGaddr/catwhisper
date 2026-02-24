# Build System

## Build Requirements

### Compiler Support

| Compiler | Minimum Version |
|----------|-----------------|
| GCC | 11+ |
| Clang | 14+ |
| MSVC | 2022 (19.30+) |

C++20 is required for:
- `std::expected` / `std::span`
- Concepts
- Ranges
- `std::format`

### System Requirements

- **Vulkan SDK**: 1.3.0 or later
- **CMake**: 3.20 or later
- **Python**: 3.8+ (for build scripts, optional)

### Vulkan SDK Installation

```bash
# Ubuntu/Debian
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.268-jammy.list \
    https://packages.lunarg.com/vulkan/1.3.268/lunarg-vulkan-1.3.268-jammy.list
sudo apt update
sudo apt install vulkan-sdk

# Fedora
sudo dnf install vulkan-tools vulkan-loader vulkan-validation-layers \
    vulkan-vulkan-profiling-tools spirv-tools spirv-headers

# macOS (via Homebrew)
brew install vulkan-headers vulkan-loader moltenvk

# Windows
# Download from https://vulkan.lunarg.com/sdk/home
```

## CMake Configuration

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)

project(catwhisper
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated vector similarity search library"
    LANGUAGES CXX
)

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(CW_BUILD_TESTS "Build unit tests" ON)
option(CW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CW_BUILD_EXAMPLES "Build example programs" ON)
option(CW_BUILD_SHARED "Build shared library" ON)
option(CW_USE_SYSTEM_VMA "Use system-provided VMA" OFF)
option(CW_ENABLE_VALIDATION "Enable Vulkan validation (debug only)" ON)
option(CW_SHADER_HOTRELOAD "Enable runtime shader compilation (debug only)" ON)

# Find dependencies
find_package(Vulkan REQUIRED)

# Vulkan Memory Allocator
if(CW_USE_SYSTEM_VMA)
    find_package(VulkanMemoryAllocator REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        VMA
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(VMA)
endif()

# Optional dependencies
if(CW_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(CW_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# Source files
file(GLOB_RECURSE CW_SOURCES
    "src/*.cpp"
)

file(GLOB_RECURSE CW_HEADERS
    "include/*.hpp"
)

# Create library
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
        VulkanMemoryAllocator::VulkanMemoryAllocator
)

# Compile definitions
target_compile_definitions(catwhisper
    PRIVATE
        $<$<CONFIG:Debug>:CW_DEBUG>
        $<$<AND:$<CONFIG:Debug>,${CW_ENABLE_VALIDATION}>:CW_VULKAN_VALIDATION>
        $<$<AND:$<CONFIG:Debug>,${CW_SHADER_HOTRELOAD}>:CW_SHADER_HOTRELOAD>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    # MoltenVK compatibility
    target_link_libraries(catwhisper PRIVATE "-framework Metal" "-framework QuartzCore")
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(catwhisper PRIVATE VK_USE_PLATFORM_XCB_KHR)
    find_package(X11 REQUIRED)
    target_link_libraries(catwhisper PRIVATE ${X11_LIBRARIES})
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(catwhisper PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Werror=return-type
    )
elseif(MSVC)
    target_compile_options(catwhisper PRIVATE
        /W4
        /wd4100  # unreferenced formal parameter
        /WX      # warnings as errors for return type issues
    )
endif()

# Shader compilation
add_subdirectory(shaders)

# Tests
if(CW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(CW_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Examples
if(CW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS catwhisper
    EXPORT catwhisper-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/catwhisper
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/catwhisperConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/catwhisperConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/catwhisper
)

install(EXPORT catwhisper-targets
    FILE catwhisperTargets.cmake
    NAMESPACE catwhisper::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/catwhisper
)

# Generate package config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/catwhisperConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/catwhisperConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/catwhisper
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/catwhisperConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
```

### Shader Compilation (shaders/CMakeLists.txt)

```cmake
find_program(GLSLANG_VALIDATOR glslangValidator)
find_program(SPIRV_OPT spirv-opt)

if(NOT GLSLANG_VALIDATOR)
    message(WARNING "glslangValidator not found, shaders won't be compiled")
    return()
endif()

file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.comp")

set(SHADER_OUTPUTS "")

foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    
    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER} -o ${OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
    )
    
    # Optimize if spirv-opt is available
    if(SPIRV_OPT)
        set(OPT_OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}_opt.spv")
        add_custom_command(
            OUTPUT ${OPT_OUTPUT}
            COMMAND ${SPIRV_OPT} -O ${OUTPUT} -o ${OPT_OUTPUT}
            DEPENDS ${OUTPUT}
            COMMENT "Optimizing shader ${SHADER_NAME}"
        )
        list(APPEND SHADER_OUTPUTS ${OPT_OUTPUT})
    else()
        list(APPEND SHADER_OUTPUTS ${OUTPUT})
    endif()
endforeach()

add_custom_target(catwhisper_shaders ALL DEPENDS ${SHADER_OUTPUTS})

# Embed shaders in library (optional, for single-binary deployment)
option(CW_EMBED_SHADERS "Embed shaders in the library binary" OFF)

if(CW_EMBED_SHADERS)
    # Generate C++ header with embedded SPIR-V
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/generated/embedded_shaders.hpp"
        COMMAND ${PYTHON_EXECUTABLE} 
            "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.py"
            "${CMAKE_BINARY_DIR}/shaders"
            "${CMAKE_BINARY_DIR}/generated/embeded_shaders.hpp"
        DEPENDS catwhisper_shaders
    )
    
    target_sources(catwhisper PRIVATE
        "${CMAKE_BINARY_DIR}/generated/embedded_shaders.hpp"
    )
    target_compile_definitions(catwhisper PRIVATE CW_EMBED_SHADERS)
endif()
```

### embed_shaders.py

```python
#!/usr/bin/env python3
"""Generate C++ header with embedded SPIR-V shaders."""

import sys
import os
from pathlib import Path

def generate_header(shader_dir: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("// Auto-generated file - do not edit\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("namespace cw::shaders {\n\n")
        
        for spv_file in sorted(shader_dir.glob("*.spv")):
            shader_name = spv_file.stem
            
            with open(spv_file, 'rb') as sf:
                data = sf.read()
            
            f.write(f"inline const uint32_t {shader_name}[] = {{\n")
            
            # Convert to uint32_t array
            words = []
            for i in range(0, len(data), 4):
                word = int.from_bytes(data[i:i+4], 'little')
                words.append(f"    0x{word:08x}u")
            
            f.write(",\n".join(words))
            f.write("\n};\n\n")
            
            f.write(f"inline constexpr size_t {shader_name}_size = {len(data)};\n\n")
        
        f.write("} // namespace cw::shaders\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <shader_dir> <output_file>")
        sys.exit(1)
    
    shader_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    generate_header(shader_dir, output_path)
```

### Test Configuration (tests/CMakeLists.txt)

```cmake
file(GLOB_RECURSE TEST_SOURCES "unit/*.cpp" "integration/*.cpp")

add_executable(catwhisper_tests ${TEST_SOURCES})

target_link_libraries(catwhisper_tests
    PRIVATE
        catwhisper::catwhisper
        GTest::gtest
        GTest::gtest_main
)

# Include GoogleTest's CMake functions
include(GoogleTest)
gtest_discover_tests(catwhisper_tests
    PROPERTIES
        LABELS "unit"
)

# Integration tests may need GPU
set_tests_properties(
    $(catwhisper_tests | grep -E "integration|gpu")
    PROPERTIES
        LABELS "integration"
        ENVIRONMENT "CW_TEST_GPU=1"
)
```

### Benchmark Configuration (benchmarks/CMakeLists.txt)

```cmake
file(GLOB BENCHMARK_SOURCES "*.cpp")

add_executable(catwhisper_benchmarks ${BENCHMARK_SOURCES})

target_link_libraries(catwhisper_benchmarks
    PRIVATE
        catwhisper::catwhisper
        benchmark::benchmark
)

# Benchmark data files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/data/sift1m.fvecs"
    "${CMAKE_CURRENT_BINARY_DIR}/data/sift1m.fvecs"
    COPYONLY
)
```

## Build Commands

### Debug Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCW_ENABLE_VALIDATION=ON \
      -DCW_SHADER_HOTRELOAD=ON \
      ..
cmake --build . -j$(nproc)
```

### Release Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCW_ENABLE_VALIDATION=OFF \
      -DCW_EMBED_SHADERS=ON \
      ..
cmake --build . -j$(nproc)
```

### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### Windows (Ninja + Clang)

```powershell
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ ..
cmake --build . --config Release
```

### macOS

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DVulkan_DIR=/usr/local/lib/cmake/Vulkan \
      ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Running Tests

```bash
# All tests
ctest --output-on-failure

# Only unit tests (no GPU required)
ctest -L unit

# Only integration tests (requires GPU)
ctest -L integration

# Verbose output
ctest -V
```

## Running Benchmarks

```bash
# Run all benchmarks
./benchmarks/catwhisper_benchmarks

# Run specific benchmark
./benchmarks/catwhisper_benchmarks --benchmark_filter=IndexFlat

# JSON output
./benchmarks/catwhisper_benchmarks --benchmark_format=json --benchmark_out=results.json
```

## Dependencies Management

### Using vcpkg

```bash
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh

# Install dependencies
./vcpkg/vcpkg install vulkan vulkan-memory-allocator

# Configure CMake
cmake -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ..
```

### Using Conan

```bash
# conanfile.txt
[requires]
vulkan/1.3.268
vulkan-memory-allocator/3.0.1

[generators]
CMakeDeps
CMakeToolchain

[layout]
cmake_layout

# Install and build
conan install . --output-folder=build --build=missing
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build
```

## Packaging

### Debian/Ubuntu Package

```bash
# Install packaging tools
sudo apt install debhelper

# Build package
cpack -G DEB
```

### RPM Package (Fedora/RHEL)

```bash
# Build package
cpack -G RPM
```

### Windows Installer

```powershell
# Requires NSIS
cpack -G NSIS
```

### Header-only Distribution

For embedding in other projects:

```cmake
# In consuming project's CMakeLists.txt
FetchContent_Declare(
    catwhisper
    GIT_REPOSITORY https://github.com/your-org/catwhisper.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(catwhisper)

target_link_libraries(your_target PRIVATE catwhisper::catwhisper)
```

## IDE Configuration

### VS Code

```json
// .vscode/settings.json
{
    "cmake.configureOnOpen": true,
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.cppStandard": "c++20"
}
```

### CLion

CLion automatically detects CMake configuration. Enable C++20 in:
Settings → Build, Execution, Deployment → CMake → CMake options:
`-DCMAKE_CXX_STANDARD=20`

### Visual Studio

Set in Project Properties:
C/C++ → Language → C++ Language Standard → ISO C++20 Standard (/std:c++20)
