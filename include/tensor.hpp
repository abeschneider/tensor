#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <iostream>
#include <list>
#include <type_traits>
#include <tuple>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "types.hpp"
#include "index_generator.hpp"
#include "stride_generator.hpp"
#include "storage.hpp"
#include "layout.hpp"
#include "slice.hpp"

enum class TensorOrder {
    RowMajor,
    ColumnMajor
};

indices make_order(std::size_t dims, TensorOrder order) {
    if (order == TensorOrder::RowMajor) {
        return make_row_major_order(dims);
    } else {
        return make_col_major_order(dims);
    }
}

template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> tuple_to_array(std::tuple<Args...> &&tup) {
    std::array<T, sizeof...(Args)> result{};
    std::size_t i = 0;
    std::apply([&result, &i](auto &&... v) {
        ((result[i++] = v), ...);
    }, tup);

    return result;
}

template <typename T, typename Device>
class Tensor;

template <typename T, typename Device=CPU>
Tensor<T, Device> tensor(const T &value) {
    Tensor<T, Device> result{1};
    result(0) = value;

    return result;
}

template <typename T, typename Device=CPU>
Tensor<T, Device> tensor(std::initializer_list<T> values) {
    Tensor<T, Device> result({values.size()});

    std::size_t i = 0;
    for (auto value : values) {
        result(i++) = value;
    }
    return result;
}

template <typename T, typename Device=CPU>
Tensor<T, Device> tensor(std::initializer_list<std::initializer_list<T>> values) {
    std::size_t d0 = values.size();
    auto l1 = *std::begin(values);
    std::size_t d1 = l1.size();

    Tensor<T, Device> result(extent{d0, d1}, TensorOrder::RowMajor);
    std::size_t i = 0;
    for (auto row : values) {
        std::size_t j = 0;
        for (auto value : row) {
            result(i, j) = value;
            ++j;
        }
        ++i;
    }

    return result;
}

template <typename T, typename Device=CPU>
Tensor<T, Device> tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> values) {
    std::size_t d0 = values.size();
    auto l1 = *std::begin(values);
    std::size_t d1 = l1.size();
    auto l2 = *std::begin(l1);
    std::size_t d2 = l2.size();

    Tensor<T, Device> result({d0, d1, d2});
    std::size_t i = 0;
    for (auto row : values) {
        std::size_t j = 0;
        for (auto col : row) {
            std::size_t k = 0;
            for (auto value : col) {
                result(i, j, k) = value;
                ++k;
            }
            ++j;
        }
        ++i;
    }

    return result;
}

template <typename T, typename D=CPU>
class Tensor {
public:
    using NumericType = T;
    using Device = D;

    Tensor() = default;

    Tensor(extent shape, TensorOrder order):
        view_({shape, make_strides(shape, make_order(shape.size(), order))}),
        storage_(std::make_shared<Storage<T, Device>>(view_.num_elements())),
        order_(order) {}

    explicit Tensor(extent const &shape):
        Tensor(shape, TensorOrder::RowMajor) {}

    Tensor(std::shared_ptr<Storage<T, Device>> storage, const View &view):
        view_(view), storage_(storage),
        order_(TensorOrder::RowMajor) {}

    Tensor(Slice<T, Device> const &slice):
        view_(slice.view()),
        storage_(slice.storage_ptr()),
        order_(TensorOrder::RowMajor) {}

    template <typename... Args>
    T &operator ()(Args... cindices) {
        auto array_indices = tuple_to_array<index_t>(std::tuple<Args...>(cindices...));
        auto pos = calculate_offset(view_, array_indices);
        return (*storage_)[pos];
    }

    T &operator ()(const indices &index) {
        auto offset = calculate_offset(view_, index);
        return (*storage_)[offset];
    }


    T const &operator ()(const indices &index) const {
        auto offset = calculate_offset(view_, index);
        return (*storage_)[offset];
    }

    // think about ways of storing results for faster indexing
    index_generator indices() const {
        return index_generator(view_);
    }

    Slice<T, Device> operator [](index_t index) const {
        return Slice(0, index, view_, storage_);
    }

    Slice<T, Device> operator [](index_range_t index_range) const {
        return Slice(0, index_range, view_, storage_);
    }

    const View &view() const { return view_; }
    View &view() { return view_; }

    Tensor<T, Device> view(const View &new_view) const {
        return Tensor<T, Device>{storage_, new_view};
    }

    extent const &shape() const { return view_.shape; }
    index_t shape(index_t dim) const { return view_.shape[dim]; }

    std::size_t num_dims() const { return view_.shape.size(); }

    Storage<T, Device> const &storage() const { return *storage_; }
    Storage<T, Device> &storage() { return *storage_; }
    std::shared_ptr<Storage<T, Device>> storage_ptr() { return storage_; }
private:
    View view_;
    std::shared_ptr<Storage<T, Device>> storage_;
    TensorOrder order_;
};

template <typename T, typename Device>
bool operator ==(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    for (auto const &index : lhs.indices()) {
        if (lhs(index) != rhs(index)) return false;
    }

    return true;
}

#endif