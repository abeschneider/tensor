#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <range/v3/all.hpp>

#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "format.hpp"

#define ASSERT_TENSORS_EQ(expected, result) \
    ASSERT_TRUE(equals(expected, result))

TEST(TensorTestSuite, TestAddTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({2, 3, 4});

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestAddInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({2, 3, 4});

    lhs += rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestAddInvalidTensors) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({{2, 3}, {4, 5}});

    EXPECT_THROW(lhs + rhs, MismatchedDimensions);
}

TEST(TensorTestSuite, TestSubTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({0, 1, 2});

    auto result = lhs - rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestSubInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({0, 1, 2});

    lhs -= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestMulTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({2, 4, 6});

    auto result = lhs * rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestMulInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({2, 4, 6});

    lhs *= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestDivTensors1) {
    auto lhs = tensor({2, 4, 6});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({1, 2, 3});

    auto result = lhs / rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestDivInplaceTensors1) {
    auto lhs = tensor({2, 4, 6});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({1, 2, 3});

    lhs /= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestAddTensors2) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    auto rhs = tensor({
        {1, 1, 1},
        {0, 0, 0},
        {1 ,1, 1}
    });

    auto expected = tensor({
        {2, 3, 4},
        {4, 5, 6},
        {8, 9, 10}
    });

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestAddInplaceTensors2) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    auto rhs = tensor({
        {1, 1, 1},
        {0, 0, 0},
        {1 ,1, 1}
    });

    auto expected = tensor({
        {2, 3, 4},
        {4, 5, 6},
        {8, 9, 10}
    });

    lhs += rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestSubTensors2) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    auto rhs = tensor({
        {1, 1, 1},
        {0, 0, 0},
        {1 ,1, 1}
    });

    auto expected = tensor({
        {0, 1, 2},
        {4, 5, 6},
        {6, 7, 8}
    });

    auto result = lhs - rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestSubInplaceTensors2) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    auto rhs = tensor({
        {1, 1, 1},
        {0, 0, 0},
        {1 ,1, 1}
    });

    auto expected = tensor({
        {0, 1, 2},
        {4, 5, 6},
        {6, 7, 8}
    });

    lhs -= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorTestSuite, TestDotProduct1) {
    auto lhs = tensor({1, 1, 1, 1, 1});
    auto rhs = tensor({2, 2, 2, 2, 2});

    auto expected = tensor({10});
    auto result = dot(lhs, rhs);

    ASSERT_EQ(expected.shape(), result.shape());
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestDotProduct2) {
    auto lhs = tensor({{1, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1}});

    auto rhs = tensor({2, 2, 2, 2, 2});

    auto expected = tensor({10, 10, 10});
    auto result = dot(lhs, rhs);

    // 3x5 * 5 => 3
    ASSERT_EQ(expected.shape(), result.shape());
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestDotProduct3) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    });

    auto rhs = tensor({
        {2, 2, 2, 2},
        {2, 2, 2, 2},
        {2, 2, 2, 2}
    });

    auto expected = tensor({
        {12, 12, 12, 12},
        {30, 30, 30, 30},
        {48, 48, 48, 48},
        {66, 66, 66, 66}
    });

    auto result = dot(lhs, rhs);

    // 4x3 * 3x4 => 4x4
    ASSERT_EQ(expected.shape(), result.shape());
    ASSERT_TENSORS_EQ(expected, result);
}