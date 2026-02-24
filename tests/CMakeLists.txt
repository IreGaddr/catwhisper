find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

file(GLOB UNIT_TESTS "unit/*.cpp")
add_executable(catwhisper_tests ${UNIT_TESTS})

target_link_libraries(catwhisper_tests
    PRIVATE
        catwhisper::catwhisper
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(catwhisper_tests)
