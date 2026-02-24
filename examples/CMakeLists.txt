add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example PRIVATE catwhisper::catwhisper)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE catwhisper::catwhisper)
