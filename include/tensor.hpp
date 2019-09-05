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

inline indices make_order(std::size_t dims, TensorOrder order) {
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

template <typename T, typename Device=CPU>
Tensor<T, Device> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> values) {
    std::size_t d0 = values.size();
    auto l1 = *std::begin(values);
    std::size_t d1 = l1.size();
    auto l2 = *std::begin(l1);
    std::size_t d2 = l2.size();
    auto l3 = *std::begin(l2);
    std::size_t d3 = l3.size();

    Tensor<T, Device> result({d0, d1, d2, d3});
    std::size_t b = 0;
    for (auto batch : values) {
        std::size_t i = 0;
        for (auto row : batch) {
            std::size_t j = 0;
            for (auto col : row) {
                std::size_t k = 0;
                for (auto value : col) {
                    result(b, i, j, k) = value;
                    ++k;
                }
                ++j;
            }
            ++i;
        }
        ++b;
    }

    return result;
}

template <typename T, typename Device>
struct ElementTensor;

/**
 * \class Tensor tensor.hpp include/tensor.hpp
 * \brief The Tensor class holds a pointer to data held in storage and a view of that data.
 * \tparam T The type of the Tensor (e.g. ``int``, ``float``, ``double``, etc.)
 * \tparam D The device the data is stored on (e.g. `CPU`, `GPU`, etc.)
 */
template <typename T, typename D=CPU>
class Tensor {
public:
    using NumericType = T;
    using Device = D;

    /**
     * \brief Copy constructor
     * \param Tensor to copy
     */
    Tensor(Tensor const &t):
        view_(t.view_), storage_(t.storage_), order_(t.order_) {}

    /**
     * \brief Constructs a Tensor with given a shape and order
     * \param shape The extent of the Tensor in each dimension
     * \param order Either TensorOrder::RowMajor or TensorOrder::ColumnMajor
     */
    Tensor(extent shape, TensorOrder order):
        view_(shape, make_order(shape.size(), order)),
        storage_(std::make_shared<Storage<T, Device>>(view_.num_elements())),
        order_(order) {}

    /**
     * \brief Constructs a Tensor with a given shape
     * \param shape The extent of the Tensor in each dimension
     */
    explicit Tensor(extent const &shape):
        Tensor(shape, TensorOrder::RowMajor) {}

    /**
     * \brief Constructs a Tensor with the given `storage` and `view`
     * \param storage The storage containing the Tensor's data
     * \param view The view of the storage
     */
    Tensor(std::shared_ptr<Storage<T, Device>> storage, const View &view):
        view_(view), storage_(storage),
        order_(TensorOrder::RowMajor) {}

    /**
     * \brief Constructs a Tensor from a Slice
     * \param slice The slice of another Tensor
     */
    Tensor(Slice<T, Device> const &slice):
        view_(slice.view()),
        storage_(slice.storage_ptr()),
        order_(TensorOrder::RowMajor) {}

    /**
     * \brief Index operator
     * \param cindices A sequence of index's used to calculate an offset
     * \return The value in the tensor for the given index
     */
    template <typename... Args>
    T &operator ()(Args... cindices) {
        auto array_indices = tuple_to_array<index_t>(std::tuple<Args...>(cindices...));
        auto pos = calculate_offset(view_, array_indices);
        return (*storage_)[pos];
    }

    /**
     * \brief Index operator
     * \param cindices A sequence of index's used to calculate an offset
     * \return The value in the tensor for the given index
     */
    template <typename... Args>
    T &operator ()(Args... cindices) const {
        auto array_indices = tuple_to_array<index_t>(std::tuple<Args...>(cindices...));
        auto pos = calculate_offset(view_, array_indices);
        return (*storage_)[pos];
    }


    /**
     * \brief Index operator
     * \param cindices A sequence of index's used to calculate an offset
     * \return The value in the tensor for the given index
     */
    T &operator ()(indices const &index) {
        auto offset = calculate_offset(view_, index);
        return (*storage_)[offset];
    }


    /**
     * \brief Index operator
     * \param cindices A sequence of index's used to calculate an offset
     * \return The value in the tensor for the given index
     */
    T const &operator ()(indices const &index) const {
        auto offset = calculate_offset(view_, index);
        return (*storage_)[offset];
    }

    // think about ways of storing results for faster indexing
    index_generator indices() const {
        return index_generator(view_);
    }

    /**
     * \brief Slice operator
     * \param index Index to set first dimension of Slice to
     * \return A Slice with the same storage as this Tensor
     */
    Slice<T, Device> operator [](index_t index) const {
        return Slice(0, index, view_, storage_);
    }

    /**
     * \brief Slice operator
     * \param index_index Range to set first dimension of Slice to
     * \return A Slice with the same storage as this Tensor
     */
    Slice<T, Device> operator [](index_range_t index_range) const {
        return Slice(0, index_range, view_, storage_);
    }

    /**
     * \brief Returns the Tensor's view
     * \return View
     */
    const View &view() const { return view_; }

    /**
     * \brief Returns the Tensor's view
     * \return View
     */
    View &view() { return view_; }

    /**
     * \brief Creates a new Tensor using the same storage but
     *        a new view
     * \return Tensor
     */
    Tensor<T, Device> view(const View &new_view) const {
        return Tensor<T, Device>{storage_, new_view};
    }

    /**
     * \brief Returns shape of tensor (from view)
     * \return extent
     */
    extent const &shape() const { return view_.shape; }

    /**
     * \brief Returns shape of tensor (from view)
     * \return extent
     */
    index_t shape(index_t dim) const { return view_.shape[dim]; }

    std::size_t num_dims() const { return view_.shape.size(); }

    Storage<T, Device> const &storage() const { return *storage_; }
    Storage<T, Device> &storage() { return *storage_; }
    std::shared_ptr<Storage<T, Device>> storage_ptr() { return storage_; }

    /**
     * \brief Returns true if Tensor is contiguous
     * \return
     */
    bool contiguous() const {
        auto contiguous_order = make_row_major_order(num_dims());
        return view_.order == contiguous_order;
    }

    ElementTensor<T, Device> el() {
        return ElementTensor<T, Device>{*this};
    }
private:
    View view_;
    std::shared_ptr<Storage<T, Device>> storage_;

    // row-major or column major
    TensorOrder order_;

    // if true, tensor elements can be changed
    bool mutable_;
};

/**
 * @brief Provides a wrapper around Tensor that causes elementwise operations to
 *        be used. Do not use this struct directly, but instead use `Tensor.el()`
 * @tparam T
 * @tparam Device
 */
template <typename T, typename Device>
struct ElementTensor {
    Tensor<T, Device> &tensor;
};

template <typename T, typename Device>
index_t num_dims(Tensor<T, Device> const &tensor) {
    return tensor.shape().size();
}

template <typename T, typename Device>
index_t num_elements(Tensor<T, Device> const &tensor) {
    return num_elements(tensor.shape());
}

template <typename T, typename Device>
Tensor<T, Device> make_contiguous(Tensor<T, Device> const &t) {
    if (t.contiguous()) return t;
    return Tensor<T, Device>(t);
}

template <typename T, typename Device>
Tensor<T, Device> copy(Tensor<T, Device> const &tensor) {
    auto new_storage = std::make_shared<Storage<T, Device>>(tensor.view().num_elements());
    for (index_t i = 0; i < tensor.view().num_elements(); i++) {
        (*new_storage)[i] = tensor.storage()[i];
    }

    return Tensor<T, Device>(new_storage, tensor.view());
}

// template <typename T, typename Device>
// Tensor<T, Device> copy(Tensor<T, Device> const &tensor, View view) {
//     auto new_storage = std::make_shared<Storage<T, Device>>(view.num_elements());
//     for (index_t i = 0; i < view.num_elements(); i++) {
//         (*new_storage)[i] = tensor.storage()[i];
//     }
//     return Tensor<T, Device>(new_storage, view);
// }

#endif