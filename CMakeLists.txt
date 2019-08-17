cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")


# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} include)

find_package(fmt)

add_executable(TensorTests
    test/tests.cpp
    test/test_indexing.cpp
    test/test_storage.cpp
    test/test_tensor.cpp
    test/test_tensor_ops.cpp
)
target_link_libraries(TensorTests ${GTEST_LIBRARIES} pthread fmt::fmt)
set_target_properties(TensorTests PROPERTIES CXX_STANDARD 17)

include(GoogleTest)
gtest_discover_tests(TensorTests)