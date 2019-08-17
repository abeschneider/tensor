#include <gtest/gtest.h>
#include <fmt/format.h>

#include "layout.hpp"

// TEST(LayoutTestSuite, TestView) {
//     View test{{2, 3}, {1, 2}, {0, 1}, {0, 0}};

//     {
//         indices index{0, 0};
//         fmt::print("offset = {}\n", test.calculate_offset(std::begin(index), std::end(index)));
//     }

//     {
//         indices index{0, 1};
//         fmt::print("offset = {}\n", test.calculate_offset(std::begin(index), std::end(index)));
//     }

//     {
//         indices index{1, 0};
//         fmt::print("offset = {}\n", test.calculate_offset(std::begin(index), std::end(index)));
//     }

//     View test2{{2, 3}, {1, 2}, {0, 1}, {1, 1}};

//     {
//         indices index{0, 0};
//         fmt::print("offset = {}\n", test2.calculate_offset(std::begin(index), std::end(index)));
//     }

//     {
//         indices index{0, 1};
//         fmt::print("offset = {}\n", test2.calculate_offset(std::begin(index), std::end(index)));
//     }

//     {
//         indices index{1, 0};
//         fmt::print("offset = {}\n", test2.calculate_offset(std::begin(index), std::end(index)));
//     }

//     fmt::print("singleton test\n");
//     View test3{{2, 1, 3}, {1, 1, 2}, {0, 1, 2}, {0, 0, 0}};
//     {
//         indices index{1, 2};
//         fmt::print("offset = {}\n", test3.calculate_offset(std::begin(index), std::end(index)));
//     }
// }

// #include "tensor.hpp"
/*
TEST(TensorTestSuite, MakeCPUTensor0d) {
    auto t = tensor(42);
    ASSERT_EQ(1, t.shape().size());
    ASSERT_EQ(1, t.shape()[0]);
}

TEST(TensorTestSuite, MakeCPUTensor1d) {
    auto t = tensor({1, 2, 3});
    ASSERT_EQ(1, t.shape().size());
    ASSERT_EQ(3, t.shape()[0]);

    ASSERT_EQ(1, t(0));
    ASSERT_EQ(2, t(1));
    ASSERT_EQ(3, t(2));
}

TEST(TensorTestSuite, MakeCPUTensor2d) {
    auto t = tensor({
        {1, 2, 3},
        {4, 5, 6}
    });
    ASSERT_EQ(2, t.shape().size());
    ASSERT_EQ(2, t.shape()[0]);
    ASSERT_EQ(3, t.shape()[1]);

    // row 0
    ASSERT_EQ(1, t(0, 0));
    ASSERT_EQ(2, t(0, 1));
    ASSERT_EQ(3, t(0, 2));

    // row 1
    ASSERT_EQ(4, t(1, 0));
    ASSERT_EQ(5, t(1, 1));
    ASSERT_EQ(6, t(1, 2));
}

TEST(TensorTestSuite, MakeCPUTensor3d) {
    auto t = tensor({
        {
            {1, 2, 3},
            {4, 5, 6}
        },
        {
            {7, 8, 9},
            {10, 11, 12}
        }
    });
    ASSERT_EQ(3, t.shape().size());
    ASSERT_EQ(2, t.shape()[0]);
    ASSERT_EQ(2, t.shape()[1]);
    ASSERT_EQ(3, t.shape()[2]);

    ASSERT_EQ(1, t(0, 0, 0));
    ASSERT_EQ(2, t(0, 0, 1));
    ASSERT_EQ(3, t(0, 0, 2));

    ASSERT_EQ(4, t(0, 1, 0));
    ASSERT_EQ(5, t(0, 1, 1));
    ASSERT_EQ(6, t(0, 1, 2));

    ASSERT_EQ(7, t(1, 0, 0));
    ASSERT_EQ(8, t(1, 0, 1));
    ASSERT_EQ(9, t(1, 0, 2));

    ASSERT_EQ(10, t(1, 1, 0));
    ASSERT_EQ(11, t(1, 1, 1));
    ASSERT_EQ(12, t(1, 1, 2));
}

TEST(TensorTestSuite, TensorIndex1d) {
    TensorIndex index(1, {10});

    for (auto i = 0; i < 10; i++) {
        ASSERT_EQ(i, index[0]);
        ++index;
    }

    // the index will wrap back to 0
    ASSERT_EQ(0, index[0]);
}

TEST(TensorTestSuite, TensorIndex2d) {
    TensorIndex index(2, {5, 5});

    for (auto i = 0; i < 5; i++) {
        for (auto j = 0; j < 5; j++) {
            ASSERT_EQ(i, index[0]);
            ASSERT_EQ(j, index[1]);
            ++index;
        }
    }

    ASSERT_EQ(0, index[0]);
    ASSERT_EQ(0, index[1]);
}

TEST(TensorTestSuite, TensorIndex3d) {
    TensorIndex index(3, {5, 5, 5});

    for (auto i = 0; i < 5; i++) {
        for (auto j = 0; j < 5; j++) {
            for (auto k = 0; k < 5; k++) {
                ASSERT_EQ(i, index[0]);
                ASSERT_EQ(j, index[1]);
                ASSERT_EQ(k, index[2]);
                ++index;
            }
        }
    }

    ASSERT_EQ(0, index[0]);
    ASSERT_EQ(0, index[1]);
    ASSERT_EQ(0, index[2]);
}

TEST(TensorTestSuite, IteratorCPUTensor1d) {
    auto t = tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // test reading
    int i = 0;
    for (auto const &el : t) {
        ASSERT_EQ(i++, el);
    }

    // test writing
    for (auto &el : t) {
        el = 0;
    }

    // verify write
    for (auto const &el : t) {
        ASSERT_EQ(0, el);
    }
}

TEST(TensorTestSuite, IteratorCPUTensor1dView) {
    auto t = tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // create a view of t
    auto t2 = t.view({{8}, {1}, t.view().strides, t.view().order});

    ASSERT_EQ(8, t2.shape()[0]);

    // test read
    int i = 1;
    for (auto const &el : t2) {
        ASSERT_EQ(i++, el);
    }

    // test write
    for (auto &el : t2) {
        el = -1;
    }

    // verify write
    for (auto const &el : t2) {
        ASSERT_EQ(-1, el);
    }

    // verify we didn't write outside the view
    ASSERT_EQ(0, t(0));
    ASSERT_EQ(9, t(9));

    // verify we wrote inside the view
    for (int i = 1; i < 9; i++) {
        ASSERT_EQ(-1, t(i));
    }
}

TEST(TensorTestSuite, IteratorCPUTensor2d) {
    auto t = tensor({
        {1, 4, 7},
        {2, 5, 8},
        {3, 6, 9}
    });

    // expected order for values to be read
    int expected[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    // test reading
    int i = 0;
    for (auto const &el : t) {
        ASSERT_EQ(expected[i++], el);
    }

    // test writing
    for (auto &el : t) {
        el = 0;
    }

    // verify writing
    for (auto const &el : t) {
        ASSERT_EQ(el, 0);
    }
}

TEST(TensorTestSuite, IteratorCPUTensor2dView) {
    auto t = tensor({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });

    // create a view of t
    auto t2 = t.view({{2, 2}, {1, 1}, t.view().strides, t.view().order});

    ASSERT_EQ(2, t2.shape()[0]);
    ASSERT_EQ(2, t2.shape()[1]);

    int i = 0;
    int values[] = {5, 6, 8, 9};
    for (auto const &el : t2) {
        ASSERT_EQ(values[i++], el);
    }

    // test write
    for (auto &el : t2) {
        el = -1;
    }

    // verify write
    for (auto const &el : t2) {
        ASSERT_EQ(-1, el);
    }

    // verify we didn't write outside the view
    ASSERT_EQ(1, t(0, 0));
    ASSERT_EQ(2, t(0, 1));
    ASSERT_EQ(3, t(0, 2));
    ASSERT_EQ(4, t(1, 0));
    ASSERT_EQ(7, t(2, 0));

    // verify we did write inside the view
    ASSERT_EQ(-1, t(1, 1));
    ASSERT_EQ(-1, t(1, 2));
    ASSERT_EQ(-1, t(2, 1));
    ASSERT_EQ(-1, t(2, 2));
}

TEST(TensorTestSuite, TranposeCPUTensor2dWithViewOffset) {
    auto t = tensor({
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11},
        {12, 13, 14, 15}
    });

    auto t2 = t.view({{3, 2}, {0, 1}, t.view().strides, t.view().order});

    ASSERT_EQ(1, t2(0, 0));
    ASSERT_EQ(2, t2(0, 1));
    ASSERT_EQ(5, t2(1, 0));
    ASSERT_EQ(6, t2(1, 1));
    ASSERT_EQ(9, t2(2, 0));
    ASSERT_EQ(10, t2(2, 1));

    auto t3 = transpose(t2);

    ASSERT_EQ(1, t3(0, 0));
    ASSERT_EQ(5, t3(0, 1));
    ASSERT_EQ(9, t3(0, 2));
    ASSERT_EQ(2, t3(1, 0));
    ASSERT_EQ(6, t3(1, 1));
    ASSERT_EQ(10, t3(1, 2));
}

TEST(TensorTestSuite, SliceCPUTensor2d) {
    auto t = tensor({
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8}
    });

    // [3, 4, 5]
    Tensor<int> t2 = t[1];

    // verify the values match
    ASSERT_EQ(3, t2(0));
    ASSERT_EQ(4, t2(1));
    ASSERT_EQ(5, t2(2));
}

TEST(TensorTestSuite, SliceCPUTensor3d) {
      auto t = tensor({
        {
            {0, 1, 2},
            {3, 4, 5},
        },
        {
            {6, 7, 8},
            {9, 10, 11}
        }
    });

    // [ [ 6,  7,  8]
    //   [ 9, 10, 11] ]
    Tensor<int> t2 = t[1];

    ASSERT_EQ(6, t2(0, 0));
    ASSERT_EQ(7, t2(0, 1));
    ASSERT_EQ(8, t2(0, 2));
    ASSERT_EQ(9, t2(1, 0));
    ASSERT_EQ(10, t2(1, 1));
    ASSERT_EQ(11, t2(1, 2));

    // [9, 10, 11]
    Tensor<int> t3 = t2[1];

    ASSERT_EQ(9, t3(0));
    ASSERT_EQ(10, t3(1));
    ASSERT_EQ(11, t3(2));
}

TEST(TensorTestSuite, MultiSliceCPUTensor3d) {
      auto t = tensor({
        {
            {0, 1, 2},
            {3, 4, 5},
        },
        {
            {6, 7, 8},
            {9, 10, 11}
        }
    });

    // [10]
    Tensor<int> t2 = t[1][1][1];
    ASSERT_EQ(10, t2(0));

    // verify that we can still slice the single value
    // [10]
    Tensor<int> t3 = t2[0];
    ASSERT_EQ(10, t3(0));
}

TEST(TensorTestSuite, RangeSliceCPUTensor3d) {
      auto t = tensor({
        {
            {0, 1, 2},
            {3, 4, 5},
        },
        {
            {6, 7, 8},
            {9, 10, 11}
        }
    });

    // [ [ 0, 1,  2]
    //   [ 3, 4,  5] ]
    //
    // auto slice = t[0];
    Tensor<int> t2 = t[0]; //slice; //[{0, 2}][{0, 3}];
    fmt::print("\nt2 = {}\n", t2);

    ASSERT_EQ(0, t2(0, 0));
    ASSERT_EQ(1, t2(0, 1));
    ASSERT_EQ(2, t2(0, 2));
    ASSERT_EQ(3, t2(1, 0));
    ASSERT_EQ(4, t2(1, 1));
    ASSERT_EQ(5, t2(1, 2));

    // [ [1, 2],
    //   [4, 5] ]
    Tensor<int> t3 = t[0][{0, 2}][{1, 3}];
    fmt::print("\nt2 = {}\n", t3);

    ASSERT_EQ(1, t3(0, 0));
    ASSERT_EQ(2, t3(0, 1));
    ASSERT_EQ(4, t3(1, 0));
    ASSERT_EQ(5, t3(1, 1));

    // [1  4]
    Tensor<int> t4 = t[0][{0, 2}][{1, 2}];
    fmt::print("\nt4 = {}\n", t4);
    ASSERT_EQ(1, t4(0));
    ASSERT_EQ(4, t4(1));
}*/



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}