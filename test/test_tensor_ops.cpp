#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <range/v3/all.hpp>

#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "format.hpp"

#define ASSERT_TENSORS_EQ(expected, result) \
    ASSERT_TRUE(equals(expected, result))

template <typename T, typename Device>
void fill_tensor(Tensor<T, Device> &tensor) {
    index_t i = 0;
    for (auto &index : index_generator(tensor.shape())) {
        tensor(index) = i++;
    }
}


TEST(TensorTestSuite, TestReshape) {
    auto t = Tensor<int>({2, 3, 4});
    fill_tensor(t);

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

TEST(TensorTestSuite, TestBroadcast1) {
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

TEST(TensorTestSuite, TestBroadcast2) {
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

TEST(TensorTestSuite, TestBroadcast3) {
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

TEST(TensorTestSuite, TestAddTensors1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1, 1, 1});
    auto expected = tensor({2, 3, 4});

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestAddTensorsWithBroadcast1) {
    auto lhs = tensor({1, 2, 3});
    auto rhs = tensor({1});
    auto expected = tensor({2, 3, 4});

    auto result = lhs + rhs;
    ASSERT_TENSORS_EQ(expected, result);
}

TEST(TensorTestSuite, TestAddTensorsWithBroadcast2) {
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

TEST(TensorTestSuite, TestAddTensorsWithBroadcast3) {
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

TEST(TensorTestSuite, TestAddTensorsWithBroadcast4) {
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

    EXPECT_THROW(lhs + rhs, CannotBroadcast);
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