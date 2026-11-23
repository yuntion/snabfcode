cmake_minimum_required(VERSION 3.20)
project(snabf_native_search LANGUAGES CXX)

# Standalone proposal tool, deliberately not a product emitter.
option(BUILD_TESTING "Build compiler tests" OFF)
add_subdirectory(../../compiler compiler EXCLUDE_FROM_ALL)
add_executable(snabf-n2-propose NativeProposal.cpp)
target_link_libraries(snabf-n2-propose PRIVATE snabf_native_arithmetic)
