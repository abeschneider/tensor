#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <range/v3/all.hpp>

#include "storage.hpp"

template <typename T>
class StorageTestSuite: public ::testing::Test {};

using StorageTypes = ::testing::Types<int, unsigned, float, double>;
TYPED_TEST_CASE(StorageTestSuite, StorageTypes);

TYPED_TEST(StorageTestSuite, TestCPUCreate) {
    Storage<TypeParam, CPU> storage(10);

    ASSERT_EQ(10, storage.size());

    for (auto e : storage) {
        ASSERT_EQ(0, e);
    }

    ranges::fill(storage, 1);

    for (auto e : storage) {
        ASSERT_EQ(1, e);
    }

    for (index_t i = 0; i < storage.size(); i++) {
        storage[i] = 2;
    }

    for (index_t i = 0; i < storage.size(); i++) {
        ASSERT_EQ(2, storage[i]);
    }
}