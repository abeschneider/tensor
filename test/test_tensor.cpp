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

TEST(TensorTestSuite, TestMakeRowMajorTensor2d) {
    extent shape{3, 3};
    Tensor<int> t(shape, TensorOrder::RowMajor);

    fill_tensor(t);

    int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    // verify storage contents are row-major
    for (index_t i = 0; i < t.storage().size(); i++) {
        ASSERT_EQ(expected[i], t.storage()[i]);
    }
}

TEST(TensorTestSuite, TestMakeColMajorTensor2d) {
    Tensor<int> t({3, 3}, TensorOrder::ColumnMajor);

    fill_tensor(t);

    int expected[] = {0, 3, 6, 1, 4, 7, 2, 5, 8};

    // verify storage contents are column-major
    for (index_t i = 0; i < t.storage().size(); i++) {
        ASSERT_EQ(expected[i], t.storage()[i]);
    }
}

// tests that even though t1 and t2 store their contents in a different ordering,
// the same indexing can be used
TEST(TensorTestSuite, TestOrdering) {
    Tensor<int> t1({3, 5}, TensorOrder::RowMajor);
    Tensor<int> t2({3, 5}, TensorOrder::ColumnMajor);

    int count = 0;
    for (auto &index : index_generator(t1.shape(), t1.view().strides)) {
        t1(index) = count;
        t2(index) = count;
        ++count;
    }

    for (auto &index : index_generator(t1.shape(), t1.view().strides)) {
        ASSERT_EQ(t1(index), t2(index));
    }
}

TEST(TensorTestSuite, TestSliceTensor) {
    Tensor<int> t({3, 3});

    {
        auto slice = t[0];
        ASSERT_EQ((extent{1, 3}), slice.view().shape);
        ASSERT_EQ(0, slice.view().offset[0]);

        auto slice2 = slice[0];
        ASSERT_EQ((extent{1, 1}), slice2.view().shape);
        ASSERT_EQ(0, slice.view().offset[0]);
    }

    {
        auto slice = t[1];
        ASSERT_EQ((extent{1, 3}), slice.view().shape);
        ASSERT_EQ(1, slice.view().offset[0]);

        auto slice2 = slice[2];
        ASSERT_EQ((extent{1, 1}), slice2.view().shape);
        ASSERT_EQ(2, slice2.view().offset[1]);
    }
}

TEST(TensorTestSuite, TestSliceRangeTensor) {
    Tensor<int> t({5, 3});
    fill_tensor(t);

    {
        Tensor<int> t2 = t[{0, 3}];

        auto expected = tensor({
            {0, 1, 2},
            {3, 4, 5},
            {6, 7, 8}
        });


        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        Tensor<int> t2 = t[{1, 4}];

        auto expected = tensor({
            {3, 4, 5},
            {6, 7, 8},
            {9, 10, 11}
        });

        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        Tensor<int> t2 = t[{1, 3}][{1, 3}];

        auto expected = tensor({
            {4, 5},
            {7, 8}
        });

        ASSERT_TENSORS_EQ(expected, t2);
    }
}

TEST(TensorTestSuite, TestMakeView) {
    auto t = Tensor<int>({5, 3});
    fill_tensor(t);

    {
        View view(t.view(), {2, 2}, {0, 0});
        auto t2 = t.view(view);

        index_t expected[] = {0, 3, 1, 4};

        index_t i = 0;
        for (auto index : t2.indices()) {
            ASSERT_EQ(expected[i++], t2(index));
        }
    }

    {
        View view(t.view(), {2, 2}, {1, 1});
        auto t2 = t.view(view);

        index_t expected[] = {4, 7, 5, 8};

        index_t i = 0;
        for (auto index : t2.indices()) {
            ASSERT_EQ(expected[i++], t2(index));
        }
    }

    {
        View view(t.view(), {5, 1}, {0, 1});
        auto t2 = t.view(view);

        index_t expected[] = {1, 4, 7, 10, 13};

        index_t i = 0;
        for (auto index : t2.indices()) {
            ASSERT_EQ(expected[i++], t2(index));
        }
    }

    {
        View view(t.view(), {1, 3}, {1, 0});
        auto t2 = t.view(view);

        index_t expected[] = {3, 4, 5};

        index_t i = 0;
        for (auto index : t2.indices()) {
            ASSERT_EQ(expected[i++], t2(index));
        }
    }
}

TEST(TensorTestSuite, TestTranspose) {
    auto t = Tensor<int>({2, 3, 4});
    fill_tensor(t);

    {
        auto t2 = transpose(t, {1, 0, 2});

        auto expected = tensor({
            {{0,   1,   2,   3},
             {12,  13,  14,  15}},

            {{4,   5,   6,   7},
             {16,  17,  18,  19}},

            {{8,   9,  10,  11},
             {20,  21, 22,  23}}
        });

        ASSERT_EQ((std::vector{4ul, 12ul, 1ul}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        auto t2 = transpose(t, {0, 2, 1});

        auto expected = tensor({
            {{ 0,  4,  8},
             { 1,  5,  9},
             { 2,  6, 10},
             { 3,  7, 11}},

            {{12, 16, 20},
             {13, 17, 21},
             {14, 18, 22},
             {15, 19, 23}}
        });

        ASSERT_EQ((std::vector{12ul, 1ul, 4ul}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }

    {
        auto t2 = transpose(t, {1, 2, 0});

        auto expected = tensor({
            {{ 0, 12},
             { 1, 13},
             { 2, 14},
             { 3, 15}},

            {{ 4, 16},
             { 5, 17},
             { 6, 18},
             { 7, 19}},

            {{ 8, 20},
             { 9, 21},
             {10, 22},
             {11, 23}}
        });

        ASSERT_EQ((std::vector{4ul, 1ul, 12ul}), t2.view().strides);
        ASSERT_TENSORS_EQ(expected, t2);
    }
}

TEST(TensorTestSuite, TestTransposeWithView) {
    auto t = Tensor<int>({2, 3, 4});
    fill_tensor(t);

    {
        Tensor<int> t2 = t[{0, 2}][{0, 3}][{0, 2}];
        auto t3 = transpose(t2, {1, 0, 2});

        auto expected = tensor({
            {{ 0,  1},
             {12, 13}},

            {{ 4,  5},
             {16, 17}},

            {{ 8,  9},
             {20, 21}},
        });

        ASSERT_EQ((std::vector{4ul, 12ul, 1ul}), t3.view().strides);
        ASSERT_TENSORS_EQ(expected, t3);
    }

    {
        Tensor<int> t2 = t[{0, 2}][{0, 3}][{1, 3}];
        auto t3 = transpose(t2, {1, 0, 2});

        auto expected = tensor({
            {{ 1,  2},
             {13, 14}},

            {{ 5,  6},
             {17, 18}},

            {{ 9, 10},
             {21, 22}},
        });

        ASSERT_EQ((std::vector{4ul, 12ul, 1ul}), t3.view().strides);
        ASSERT_TENSORS_EQ(expected, t3);
    }

}

// TODO: move to test_format.cpp
TEST(TensorTestSuite, TestFormatTensor1d) {
    Tensor<int> t({5}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[  0,   1,   2,   3,   4]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}

TEST(TensorTestSuite, TestFormatTensor2d_1) {
    Tensor<int> t({1, 5}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[[  0,   1,   2,   3,   4]]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}

TEST(TensorTestSuite, TestFormatTensor2d_2) {
    Tensor<int> t({5, 1}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[[  0],\n"
                           " [  1],\n"
                           " [  2],\n"
                           " [  3],\n"
                           " [  4]]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}

TEST(TensorTestSuite, TestFormatTensor2d_3) {
    Tensor<int> t({3, 4}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[[  0,   1,   2,   3],\n"
                           " [  4,   5,   6,   7],\n"
                           " [  8,   9,  10,  11]]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}

TEST(TensorTestSuite, TestFormatTensor3d_1) {
    Tensor<int> t({2, 3, 4}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[[[  0,   1,   2,   3],\n"
                           "  [  4,   5,   6,   7],\n"
                           "  [  8,   9,  10,  11]],\n"
                           "\n"
                           " [[ 12,  13,  14,  15],\n"
                           "  [ 16,  17,  18,  19],\n"
                           "  [ 20,  21,  22,  23]]]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}

TEST(TensorTestSuite, TestFormatTensor4d_1) {
    Tensor<int> t({2, 2, 3, 4}, TensorOrder::RowMajor);

    fill_tensor(t);

    std::string expected = "[[[[  0,   1,   2,   3],\n"
                           "   [  4,   5,   6,   7],\n"
                           "   [  8,   9,  10,  11]],\n"
                           "\n"
                           "  [[ 12,  13,  14,  15],\n"
                           "   [ 16,  17,  18,  19],\n"
                           "   [ 20,  21,  22,  23]]],\n"
                           "\n"
                           "\n"
                           " [[[ 24,  25,  26,  27],\n"
                           "   [ 28,  29,  30,  31],\n"
                           "   [ 32,  33,  34,  35]],\n"
                           "\n"
                           "  [[ 36,  37,  38,  39],\n"
                           "   [ 40,  41,  42,  43],\n"
                           "   [ 44,  45,  46,  47]]]]";

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "{}", t);
    ASSERT_EQ(fmt::to_string(buffer), expected);
}
