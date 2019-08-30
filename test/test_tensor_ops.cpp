#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <range/v3/all.hpp>

#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "format.hpp"

#define ASSERT_TENSORS_EQ(expected, result) \
    ASSERT_TRUE(equals(expected, result))

TEST(TensorOpsTestSuite, TestIsContiguous) {
    Tensor<int> t({2, 3, 4});
    iota(t);

    ASSERT_TRUE(t.contiguous());

    auto t2 = transpose(t, {1, 0, 2});
    ASSERT_FALSE(t2.contiguous());
}

TEST(TensorOpTestSuite, TestReshape1) {
    Tensor<int> t({2, 3, 4});
    iota(t);

    auto t2 = reshape(t, {6, 4});
    auto expected = tensor({
        {  0,  1,  2,  3},
        {  4,  5,  6,  7},
        {  8,  9, 10, 11},
        { 12, 13, 14, 15},
        { 16, 17, 18, 19},
        { 20, 21, 22, 23}
    });

    ASSERT_EQ((extent{6, 4}), t2.shape());
    ASSERT_TENSORS_EQ(expected, t2);

    // t was contiguous, so verify it is the same memory address
    ASSERT_EQ(t.storage_ptr(), t2.storage_ptr());
}

TEST(TensorOpTestSuite, TestReshape2) {
    Tensor<int> t({2, 3, 4});
    iota(t);

    auto t2 = transpose(t, {1, 0, 2});
    auto t3 = reshape(t2, {6, 4});

    auto expected = tensor({
        {  0,  1,  2,  3},
        {  4,  5,  6,  7},
        {  8,  9, 10, 11},
        { 12, 13, 14, 15},
        { 16, 17, 18, 19},
        { 20, 21, 22, 23}
    });

    ASSERT_EQ((extent{6, 4}), t3.shape());
    ASSERT_TENSORS_EQ(expected, t3);

    // t2 was non-contiguous, so verify it's a different memory address
    ASSERT_NE(t3.storage_ptr(), t2.storage_ptr());
}

TEST(TensorOpTestSuite, TestReshape3) {
    Tensor<int> t({2, 3, 4});
    iota(t);

    auto t2 = reshape(t, {expand, 4});
    ASSERT_EQ((extent{6, 4}), t2.shape());

    // fmt::print("got:\n{}\n", t2);
    auto expected = tensor({
        {  0,  1,  2,  3},
        {  4,  5,  6,  7},
        {  8,  9, 10, 11},
        { 12, 13, 14, 15},
        { 16, 17, 18, 19},
        { 20, 21, 22, 23}
    });

    ASSERT_TENSORS_EQ(expected, t2);
}

TEST(TensorOpsTestSuite, TestReshape) {
    auto t = Tensor<int>({2, 3, 4});
    iota(t, 0);

    {
        auto t2 = reshape(t, {24});

        auto expected = tensor({ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});

        ASSERT_EQ((std::vector{24UL}), t2.shape());
        ASSERT_EQ((std::vector{1UL}), t2.view().strides);

        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        auto t2 = reshape(t, {6, 4});

        auto expected = tensor({
            {  0,   1,   2,   3},
            {  4,   5,   6,   7},
            {  8,   9,  10,  11},
            { 12,  13,  14,  15},
            { 16,  17,  18,  19},
            { 20,  21,  22,  23}
        });

        ASSERT_EQ((std::vector{6UL, 4UL}), t2.shape());
        ASSERT_EQ((std::vector{4UL, 1UL}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        auto t2 = reshape(t, {4, 6});

        auto expected = tensor({
            {  0,   1,   2,   3,   4,   5},
            {  6,   7,   8,   9,  10,  11},
            { 12,  13,  14,  15,  16,  17},
            { 18,  19,  20,  21,  22,  23}
        });


        ASSERT_EQ((std::vector{4UL, 6UL}), t2.shape());
        ASSERT_EQ((std::vector{6UL, 1UL}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        EXPECT_THROW(reshape(t, {2, 3}), MismatchedNumberOfElements);
    }
}

TEST(TensorOpsTestSuite, TestReshapeWithInferredDimension) {
}

TEST(TensorOpsTestSuite, TestBroadcast1) {
    {
        auto t = tensor({1, 2, 3, 4});
        auto t2 = broadcast_to(t, {3, 4});

        auto expected = tensor({
            { 1, 2, 3, 4},
            { 1, 2, 3, 4},
            { 1, 2, 3, 4}
        });

        ASSERT_EQ((extent{3, 4}), t2.shape());
        ASSERT_EQ((indices{0, 1}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        auto t = tensor({1, 2, 3, 4});
        EXPECT_THROW(broadcast_to(t, {4, 3}), CannotBroadcast);
    }

    {
        auto t = tensor({3});
        auto t2 = broadcast_to(t, {3, 3});

        auto expected = tensor({
            {3, 3, 3},
            {3, 3, 3},
            {3, 3, 3}
        });


        ASSERT_EQ((extent{3, 3}), t2.shape());
        ASSERT_EQ((indices{0, 0}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
     }
}

TEST(TensorOpsTestSuite, TestBroadcast2) {
    auto t1 = tensor({
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3}
    });

    auto t2 = tensor({4, 4, 4, 4, 4});

    auto [t3, t4] = broadcast(t1, t2);

    auto expected = tensor({
        {5, 5, 5, 5, 5},
        {6, 6, 6, 6, 6},
        {7, 7, 7, 7, 7}
    });

    auto result = t3 + t4;

    ASSERT_EQ((extent{3, 5}), t3.shape());
    ASSERT_EQ((extent{3, 5}), t4.shape());
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestBroadcast3) {
    auto t1 = tensor({
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3}
    });

    auto t2 = tensor({{4, 4, 4, 4, 4}});

    auto [t3, t4] = broadcast(t1, t2);

    auto expected = tensor({
        {5, 5, 5, 5, 5},
        {6, 6, 6, 6, 6},
        {7, 7, 7, 7, 7}
    });

    auto result = t3 + t4;

    ASSERT_EQ((extent{3, 5}), t3.shape());
    ASSERT_EQ((extent{3, 5}), t4.shape());
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({2, 3, 4});

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddTensorsWithBroadcast1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1});
    auto expected = tensor({2, 3, 4});

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddTensorsWithBroadcast2) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    auto rhs = tensor({1});
    auto expected = tensor({
        {2, 3, 4},
        {5, 6, 7},
        {8, 9, 10}
    });

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddTensorsWithBroadcast3) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    });

    auto rhs = tensor({1, 2, 3});
    auto expected = tensor({
        {2, 4, 6},
        {5, 7, 9},
        {8, 10, 12},
        {11, 13, 15}
    });

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddTensorsWithBroadcast4) {
    auto lhs = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    });

    auto rhs = tensor({{1}, {2}, {3}, {4}});

    auto expected = tensor({
        {2, 3, 4},
        {6, 7, 8},
        {10, 11, 12},
        {14, 15, 16}
    });

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestAddInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({2, 3, 4});

    lhs += rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorOpsTestSuite, TestAddInvalidTensors) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({{2, 3}, {4, 5}});

    EXPECT_THROW(lhs + rhs, CannotBroadcast);
}

TEST(TensorOpsTestSuite, TestSubTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({0, 1, 2});

    auto result = lhs - rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestSubInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({0, 1, 2});

    lhs -= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorOpsTestSuite, TestMulTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({2, 4, 6});

    auto result = lhs * rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestMulInplaceTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({2, 4, 6});

    lhs *= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorOpsTestSuite, TestDivTensors1) {
    auto lhs = tensor({2, 4, 6});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({1, 2, 3});

    auto result = lhs / rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestDivInplaceTensors1) {
    auto lhs = tensor({2, 4, 6});
    auto rhs = tensor({2, 2, 2});
    auto expected = tensor({1, 2, 3});

    lhs /= rhs;
    ASSERT_TENSORS_EQ(expected, lhs);
}

TEST(TensorOpsTestSuite, TestAddTensors2) {
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

TEST(TensorOpsTestSuite, TestAddInplaceTensors2) {
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

TEST(TensorOpsTestSuite, TestSubTensors2) {
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

TEST(TensorOpsTestSuite, TestSubInplaceTensors2) {
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

TEST(TensorOpsTestSuite, TestDotProduct1) {
    auto lhs = tensor({1, 1, 1, 1, 1});
    auto rhs = tensor({2, 2, 2, 2, 2});

    auto expected = tensor({10});
    auto result = dot(lhs, rhs);

    ASSERT_EQ(expected.shape(), result.shape());
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorOpsTestSuite, TestDotProduct2) {
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

TEST(TensorOpsTestSuite, TestDotProduct3) {
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

