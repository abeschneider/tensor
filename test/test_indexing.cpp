#include <list>

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "layout.hpp"
#include "index_generator.hpp"
#include "stride_generator.hpp"

TEST(IndexTestSuite, TestMakeRowMajorOrder) {
    auto order = make_row_major_order(4);

    indices expected = {3, 2, 1, 0};
    auto expected_value = std::begin(expected);
    for (auto index : order) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestMakeColMajorOrder) {
    indices order = make_col_major_order(4);

    indices expected = {0, 1, 2, 3};
    auto expected_value = std::begin(expected);
    for (auto index : order) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestOrderedViewRowMajor) {
    extent shape{2, 3, 4};
    indices order = make_row_major_order(shape.size());
    ordered_shape_view shape_view(shape, order);

    indices expected = {4, 3, 2};
    auto expected_value = std::begin(expected);
    for (auto e : shape_view) {
        ASSERT_EQ(*expected_value++, e);
    }
}

TEST(IndexTestSuite, TestOrderedViewColMajor) {
    extent shape{2, 3, 4};
    indices order = make_col_major_order(shape.size());
    ordered_shape_view shape_view(shape, order);

    indices expected = {2, 3, 4};
    auto expected_value = std::begin(expected);
    for (auto e : shape_view) {
        ASSERT_EQ(*expected_value++, e);
    }
}

TEST(IndexTestSuite, TestStridesRowMajor1) {
    extent shape{2, 3, 4};
    indices order = make_row_major_order(shape.size());
    indices strides = make_strides(shape, order);

    ASSERT_EQ((indices{12, 4, 1}), strides);
}

TEST(IndexTestSuite, TestStridesRowMajor2) {
    {
        extent shape{4, 1};
        indices order = make_row_major_order(shape.size());
        indices strides = make_strides(shape, order);

        ASSERT_EQ((indices{1, 1}), strides);
    }

    {
        extent shape{1, 4};
        indices order = make_row_major_order(shape.size());
        indices strides = make_strides(shape, order);

        ASSERT_EQ((indices{4, 1}), strides);
    }
}

TEST(IndexTestSuite, TestStridesColMajor) {
    extent shape{2, 3, 4};
    indices order = make_col_major_order(shape.size());
    indices strides = make_strides(shape, order);

    ASSERT_EQ((indices{1, 2, 6}), strides);
}

TEST(IndexTestSuite, TestIndexGenerator1dRowMajor) {
    extent shape{5};
    indices order = make_row_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = { {0}, {1}, {2}, {3}, {4} };
 }

TEST(IndexTestSuite, TestIndexGenerator1dColMajor) {
    extent shape{5};
    indices order = make_col_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = { {0}, {1}, {2}, {3}, {4} };
    auto expected_value = std::begin(expected);

    for (auto &index : indices) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestIndexGenerator2dRowMajor) {
    extent shape{3, 3};
    indices order = make_row_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = {
        {0, 0}, {0, 1}, {0, 2},
        {1, 0}, {1, 1}, {1, 2},
        {2, 0}, {2, 1}, {2, 2}
    };
    auto expected_value = std::begin(expected);

    for (auto &index : indices) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestIndexGenerator2dColMajor) {
    extent shape{3, 3};
    indices order = make_col_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = {
        {0, 0}, {1, 0}, {2, 0},
        {0, 1}, {1, 1}, {2, 1},
        {0, 2}, {1, 2}, {2, 2}
    };
    auto expected_value = std::begin(expected);

    for (auto &index : indices) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestIndexGenerator3dRowMajor) {
    extent shape{3, 3, 3};
    indices order = make_row_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = {
        {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
        {0, 1, 0}, {0, 1, 1}, {0, 1, 2},
        {0, 2, 0}, {0, 2, 1}, {0, 2, 2},

        {1, 0, 0}, {1, 0, 1}, {1, 0, 2},
        {1, 1, 0}, {1, 1, 1}, {1, 1, 2},
        {1, 2, 0}, {1, 2, 1}, {1, 2, 2},

        {2, 0, 0}, {2, 0, 1}, {2, 0, 2},
        {2, 1, 0}, {2, 1, 1}, {2, 1, 2},
        {2, 2, 0}, {2, 2, 1}, {2, 2, 2}
    };
    auto expected_value = std::begin(expected);

    for (auto &index : indices) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestIndexGenerator3dColMajor) {
    extent shape{3, 3, 3};
    indices order = make_col_major_order(shape.size());
    indices strides = make_strides(shape, order);
    index_generator indices(shape, strides);

    std::list<extent> expected = {
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0},
        {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
        {0, 2, 0}, {1, 2, 0}, {2, 2, 0},
        {0, 0, 1}, {1, 0, 1}, {2, 0, 1},
        {0, 1, 1}, {1, 1, 1}, {2, 1, 1},
        {0, 2, 1}, {1, 2, 1}, {2, 2, 1},
        {0, 0, 2}, {1, 0, 2}, {2, 0, 2},
        {0, 1, 2}, {1, 1, 2}, {2, 1, 2},
        {0, 2, 2}, {1, 2, 2}, {2, 2, 2},
    };
    auto expected_value = std::begin(expected);

    for (auto &index : indices) {
        ASSERT_EQ(*expected_value++, index);
    }
}

TEST(IndexTestSuite, TestLayoutCalculateOffset1d) {
    extent shape{10};
    indices order = make_row_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    indices expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto expected_value = std::begin(expected);

    for (auto &index : index_generator(layout)) {
        offset_t offset = calculate_offset(layout, index);
        ASSERT_EQ(*expected_value++, offset);
    }
}

TEST(IndexTestSuite, TestLayoutCalculateOffset2dRowMajor) {
    extent shape{5, 2};
    indices order = make_row_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    indices expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto expected_value = std::begin(expected);

    for (auto &index : index_generator(layout)) {
        offset_t offset = calculate_offset(layout, index);
        ASSERT_EQ(*expected_value++, offset);
    }
}

TEST(IndexTestSuite, TestLayoutCalculateOffset2dColMajor) {
    extent shape{5, 2};
    indices order = make_col_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    indices expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto expected_value = std::begin(expected);

    for (auto &index : index_generator(layout)) {
        offset_t offset = calculate_offset(layout, index);
        ASSERT_EQ(*expected_value++, offset);
    }
}

TEST(IndexTestSuite, TestLayoutCalculateOffset3dRowMajor) {
    extent shape{2, 3, 2};
    indices order = make_row_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    indices expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto expected_value = std::begin(expected);

    for (auto &index : index_generator(layout)) {
        offset_t offset = calculate_offset(layout, index);
        ASSERT_EQ(*expected_value++, offset);
    }
}

TEST(IndexTestSuite, TestViewCalculateOffset1dRowMajor) {
    extent shape{10};
    indices order = make_row_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    extent view_shape{8};
    indices view_offset{2};
    View view(layout, view_shape, view_offset);

    indices expected = {2, 3, 4, 5, 6, 7, 8, 9};
    auto expected_value = std::begin(expected);

    for (auto &index : index_generator(view)) {
        offset_t offset = calculate_offset(view, index);
        ASSERT_EQ(*expected_value++, offset);
    }
}

TEST(IndexTestSuite, TestViewCalculateOffset2dRowMajor) {
    extent shape{3, 5};
    indices order = make_row_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    extent view_shape{2, 3};
    indices view_offset{1, 2};
    View view(layout, view_shape, view_offset);

    index_t expected[3][5] = {
        { 0,  1,  2,  3,  4},
        { 5,  6,  7,  8,  9},
        {10, 11, 12, 13, 14}
    };

    auto index = index_generator(view);
    for (index_t j = 0; j < view.shape[1]; j++) {
        for (index_t i = 0; i < view.shape[0]; i++) {
            offset_t offset = calculate_offset(view, index.read());
            auto expected_value = expected[i + view.offset[0]][j + view.offset[1]];
            ASSERT_EQ(expected_value, offset);
            index.next();
        }
    }
}

TEST(IndexTestSuite, TestViewCalculateOffset2dColMajor) {
    extent shape{3, 5};
    indices order = make_col_major_order(shape.size());
    Layout layout(shape, make_strides(shape, order));

    extent view_shape{2, 3};
    indices view_offset{1, 2};
    View view(layout, view_shape, view_offset);

    index_t expected[3][5] = {
        {0, 3, 6, 9 , 12},
        {1, 4, 7, 10, 13},
        {2, 5, 8, 11, 14}
    };

    // show view.stride is necessary for making the indices, but layout.stride is
    // necessary for storage?
    auto index = index_generator(view);
    for (index_t j = 0; j < view.shape[1]; j++) {
        for (index_t i = 0; i < view.shape[0]; i++) {
            offset_t offset = calculate_offset(view, index.read());
            auto expected_value = expected[i + view.offset[0]][j + view.offset[1]];
            ASSERT_EQ(expected_value, offset);
            index.next();
        }
    }
}

TEST(IndexTestSuite, TestViewCalculateOffsetWithSingletons) {
    extent shape{1, 1, 1, 5};
    auto order = make_row_major_order(shape.size());
    auto strides = make_strides(shape, order);

    ASSERT_EQ((indices{5, 5, 5, 1}), strides);
    View view(Layout(shape, strides));

    indices expected = {0, 1, 2, 3, 4};
    auto expected_value = std::begin(expected);

    auto index = index_generator(view);
    for (index_t i = 0; i < view.shape[3]; i++) {
        offset_t offset = calculate_offset(view, index.read());
        ASSERT_EQ(*expected_value, offset);
        index.next();
        ++expected_value;
    }

    // verify it also works using this method
    expected_value = std::begin(expected);
    for (index_t i = 0; i < view.shape[3]; i++) {
        offset_t offset = calculate_offset(view, i);
        ASSERT_EQ(*expected_value, offset);
        ++expected_value;
    }
}