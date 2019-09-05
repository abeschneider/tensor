#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdexcept>

#include <boost/optional.hpp>

#include <fmt/format.h>

#include <tensor.hpp>
#include <types.hpp>

constexpr index_t expand = std::numeric_limits<index_t>::max();

struct TensorError: public std::runtime_error {
    TensorError(std::string const &msg): std::runtime_error(msg) {}
};

struct MismatchedNumberOfElements: public TensorError {
    MismatchedNumberOfElements(std::size_t lhs, std::size_t rhs):
        TensorError(fmt::format("Number of elements are not the same: {} and {}", lhs, rhs)) {}
};

struct MismatchedDimensions: public TensorError {
    MismatchedDimensions(extent const &shape1, extent const &shape2):
        TensorError(fmt::format("Mismatched dimensions: {} and {}", shape1, shape2)) {}
};

struct CannotBroadcast: public TensorError {
    CannotBroadcast(extent const &shape1, extent const &shape2):
        TensorError(fmt::format("Cannot broadcast: {} and {}", shape1, shape2)) {}
};

struct NotEnoughDimensions: public TensorError {
    NotEnoughDimensions(extent const &shape):
        TensorError(fmt::format("Not enough dimensions: {}", shape)) {}
};

/**
 * @brief Returns a tensor fill with `0`s
 *
 * @tparam T The tensor type
 * @tparam Device=CPU The device tensor is stored on
 * @param shape The shape of the tensor
 * @return Tensor<T, Device>
 */
template <typename T, typename Device=CPU>
Tensor<T, Device> zeros(extent const &shape) {
    Tensor<T, Device> result(shape);
    fill(result, 0);
    return result;
}

/**
 * @brief Returns a tensor fill with `1`s
 *
 * @tparam T The tensor type
 * @tparam Device=CPU The device tensor is stored on
 * @param shape The shape of the tensor
 * @return Tensor<T, Device>
 */
template <typename T, typename Device=CPU>
Tensor<T, Device> ones(extent const &shape) {
    Tensor<T, Device> result(shape);
    fill(result, 0);
    return result;
}

template <typename T, typename Device>
void iota(Tensor<T, Device> &t, T start=0, T stride=1) {
    T value = start;
    fill(t, [&value, stride]() {
        T tmp = value;
        value += stride;
        return tmp;
    });
}

template <typename T, typename Device=CPU>
Tensor<T, Device> range(T start, T end, T stride=1) {
    std::size_t size = std::floor((end - start) / stride);
    Tensor<T, Device> result({size});

    iota(result, start, end, stride);
    return result;
}

/**
 * @brief Re-orders dimensions of Tensor
 *
 * @tparam T The tensor type
 * @tparam Device The device tensor is stored on
 * @param tensor The tensor to re-order
 * @param order The new order of dimensions
 * @return Tensor<T, Device>
 */
template <typename T, typename Device>
Tensor<T, Device> transpose(
    Tensor<T, Device> const &tensor,
    indices const &order)
{
    extent shape(tensor.shape().size());
    indices offset(tensor.shape().size());
    indices strides(tensor.shape().size());

    for (index_t i = 0; i < tensor.shape().size(); i++) {
        shape[i] = tensor.view().shape[order[i]];
        offset[i] = tensor.view().offset[order[i]];
        strides[i] = tensor.view().strides[order[i]];
    }

    return tensor.view({shape, offset, order, strides});
}

// TODO: move to source
inline boost::optional<index_t> get_inferred_dimension(extent const &shape) {
    boost::optional<index_t> result;

    for (index_t i = 0; i < shape.size(); i++) {
        if (shape[i] == expand) {
            if (!result) {
                result = i;
            } else {
                throw TensorError("Cannot infer more than one dimension.");
            }
        }
    }

    return result;
}

// TODO: move to source
inline void calculate_reshape(extent const &from_shape, extent &to_shape) {
    auto inferred_dimension = get_inferred_dimension(to_shape);

    if (inferred_dimension) {
        // number of elements for the new shape
        to_shape[*inferred_dimension] = 1;
        std::size_t new_size = num_elements(to_shape);

        // number of elements for the old shape
        std::size_t total_size = num_elements(from_shape);

        // calculate the size of the inferred dimension
        std::size_t inferred_size = total_size / new_size;
        to_shape[*inferred_dimension] = inferred_size;
    }
}

// TODO: make this either respect order, or take order f strides
template <typename T, typename Device>
Tensor<T, Device> reshape(Tensor<T, Device> const &tensor, extent const &shape) {
    extent new_shape = shape;
    calculate_reshape(tensor.shape(), new_shape);

    // we can only reshape if the number of elements is conserved
    if (num_elements(new_shape) != num_elements(tensor.shape())) {
        throw MismatchedNumberOfElements(num_elements(new_shape),
                                         num_elements(tensor.shape()));
    }

    auto order = make_row_major_order(new_shape.size());
    auto strides = make_strides(new_shape, order);
    auto offset = make_offset(new_shape.size());

    if (tensor.contiguous()) {
        return tensor.view({new_shape, offset, order, strides});
    }

    return copy(tensor.view({new_shape, offset, order, strides}));
}

template <typename T, typename Device>
bool is_broadcastable_to(Tensor<T, Device> const &tensor, extent const &shape) {
    if (shape.size() < tensor.shape().size()) return false;

    auto shape_it = shape.rbegin();
    auto tensor_shape_it = tensor.shape().rbegin();
    for (; tensor_shape_it != tensor.shape().rend(); ++shape_it, ++tensor_shape_it) {
        if (*shape_it != *tensor_shape_it && *tensor_shape_it != 1) return false;
    }

    return true;
}

namespace detail {

template <typename T, typename Device>
Tensor<T, Device> broadcast_to(Tensor<T, Device> const &tensor, extent const &shape) {
    extent strides(shape.size(), 0);
    auto offset = make_offset(shape.size());
    auto order = make_row_major_order(shape.size());

    std::size_t diff = shape.size() - tensor.shape().size();
    for (index_t i = 0; i < tensor.shape().size(); i++) {
        if (tensor.shape()[i] > 1) {
            strides[diff+i] = tensor.view().strides[i];
        } else {
            strides[diff+i] = 0;
        }
    }

    return tensor.view({shape, offset, order, strides});
}

} // namespace detail

template <typename T, typename Device>
Tensor<T, Device> broadcast_to(Tensor<T, Device> const &tensor, extent const &shape) {
    if (tensor.shape() == shape) return tensor;

    if (!is_broadcastable_to(tensor, shape)) {
        throw CannotBroadcast(tensor.shape(), shape);
    }

    return detail::broadcast_to(tensor, shape);
}

template <typename T, typename Device>
std::pair<Tensor<T, Device>, Tensor<T, Device>>
broadcast(Tensor<T, Device> const &t1, Tensor<T, Device> const &t2) {
    if (t1.shape() == t2.shape()) return std::make_pair(t1, t2);

    bool t1_to_t2 = is_broadcastable_to(t1, t2.shape());
    bool t2_to_t1 = is_broadcastable_to(t2, t1.shape());

    if (!t1_to_t2 && !t2_to_t1) {
        throw CannotBroadcast(t1.shape(), t2.shape());
    }

    auto result1 = t1_to_t2 ? detail::broadcast_to(t1, t2.shape()) : t1;
    auto result2 = t2_to_t1 ? detail::broadcast_to(t2, t1.shape()) : t2;
    return std::make_pair(result1, result2);
}

template <typename RT, typename T, typename Device, typename F>
Tensor<RT, Device> apply(Tensor<T, Device> const &lhs,
                         Tensor<T, Device> const &rhs,
                         F fn)
{
    if (lhs.shape() != rhs.shape()) {
        // if the dimensions don't match, try to broadcast
        auto [lhs_broadcast, rhs_broadcast] = broadcast(lhs, rhs);
        return apply<RT>(lhs_broadcast, rhs_broadcast, fn);
    }

    Tensor<RT, Device> result(lhs.shape());

    // can optimize if both lhs and rhs are contiguous
    for (auto const &index : lhs.indices()) {
        result(index) = fn(lhs(index), rhs(index));
    }

    return result;
}

template <typename T, typename Device, typename F>
void iapply(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs, F fn) {
    if (lhs.shape() != rhs.shape()) {
        throw MismatchedDimensions(lhs.shape(), rhs.shape());
    }

    // can optimize if both lhs and rhs are contiguous
    for (auto const &index : lhs.indices()) {
        lhs(index) = fn(lhs(index), rhs(index));
    }
}

template <typename RT, typename T, typename Device, typename F>
Tensor<RT, Device> apply(Tensor<T, Device> const &t, F fn) {
    Tensor<RT, Device> result(t.shape());
    for (auto const &index : t.indices()) {
        result(index) = fn(t(index));
    }

    return result;
}

template <typename T, typename Device, typename F>
void iapply(Tensor<T, Device> &t, F fn) {
    for (auto const &index : t.indices()) {
        t(index) = fn(t(index));
    }
}

/**
 * @brief Sets all the elements of a tensor to the given value
 *
 * @tparam T The tensor type
 * @tparam Device The device tensor is stored on
 * @param t The tensor to fill
 * @param value The value use
 */
template <typename T, typename Device>
void fill(Tensor<T, Device> &t, T const &value) {
    iapply(t, [value](T const &) { return value; });
}

/**
 * @brief Fills tensor using a successive calls to passed in function
 *
 * @tparam T The tensor type
 * @tparam Device The device tensor is stored on
 * @tparam F Function type
 * @param t The tensor to fill
 * @param fn The function used to fill tensor
 */
template <typename T, typename Device, typename F>
void fill(Tensor<T, Device> &t, F fn) {
    iapply(t, [fn](T const &) { return fn(); });
}

template <typename T, typename Device>
Tensor<T, Device> operator +(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop + rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator +=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    iapply(lhs, rhs, [](T const &lop, T const &rop) { return lop + rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> operator -(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop - rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator -=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    iapply(lhs, rhs, [](T const &lop, T const &rop) { return lop - rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> operator *(ElementTensor<T, Device> lhs, ElementTensor<T, Device> rhs) {
    return apply<T>(lhs.tensor, rhs.tensor, [](T const &lop, T const &rop) {
        return lop * rop;
    });
}

template <typename T, typename Device>
Tensor<T, Device> &operator *=(ElementTensor<T, Device> lhs, ElementTensor<T, Device> rhs) {
    iapply(lhs.tensor, rhs.tensor, [](T const &lop, T const &rop) { return lop * rop; });
    return lhs.tensor;
}

template <typename T, typename Device>
Tensor<T, Device> operator /(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop / rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator /=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    iapply(lhs, rhs, [](T const &lop, T const &rop) { return lop / rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> sin(Tensor<T, Device> &t) {
    return apply<T>(t, [](T const &v) { return std::sin(v); });
}

template <typename T, typename Device>
void isin(Tensor<T, Device> &t) {
    iapply(t, [](T const &v) { return std::sin(v); });
}

// move to detail
template <typename T, typename Device>
Tensor<T, Device> vector_vector_product(Tensor<T, Device> const &lhs,
                                        Tensor<T, Device> const &rhs)
{
    if (num_elements(lhs) != num_elements(rhs)) {
        throw MismatchedNumberOfElements(num_elements(lhs), num_elements(rhs));
    }

    T result(0);
    for (index_t i = 0; i < num_elements(lhs); i++) {
        result += lhs(i)*rhs(i);
    }

    return tensor({result});
}

template <typename T, typename Device>
Tensor<T, Device> matrix_vector_product(Tensor<T, Device> const &lhs,
                                        Tensor<T, Device> const &rhs)
{
    // verify inner dimensions match: (Nx1, 1)
    Tensor<T, Device> result({lhs.shape()[0]});
    fill(result, 0);

    for (index_t i = 0; i < lhs.shape()[0]; i++) {
        for (index_t j = 0; j < lhs.shape()[1]; j++) {
            result(i) += lhs(i, j)*rhs(j);
        }
    }

    return result;
}

// need batch_matrix_vector

template <typename T, typename Device>
Tensor<T, Device> matrix_matrix_product(Tensor<T, Device> const &lhs,
                                        Tensor<T, Device> const &rhs)
{
    Tensor<T, Device> result({lhs.shape()[0], rhs.shape()[1]});
    fill(result, 0);

    for (index_t i = 0; i < lhs.shape()[0]; i++) {
        for (index_t j = 0; j < lhs.shape()[1]; j++) {
            for (index_t k = 0; k < rhs.shape()[1]; k++) {
                result(i, k) += lhs(i, j)*rhs(j, k);
            }
        }
    }

    return result;
}

template <typename T, typename Device>
Tensor<T, Device> batch_matrix_matrix_product(Tensor<T, Device> const &lhs,
                                              Tensor<T, Device> const &rhs)
{
    // BxMxN * BxNxP
    Tensor<T, Device> result({lhs.shape()[0], lhs.shape()[1], rhs.shape()[2]});
    fill(result, 0);

    for (index_t b = 0; b < lhs.shape()[0]; b++) {
        for (index_t i = 0; i < lhs.shape()[1]; i++) {
            for (index_t j = 0; j < lhs.shape()[2]; j++) {
                for (index_t k = 0; k < rhs.shape()[2]; k++) {
                    result(b, i, k) += lhs(b, i, j)*rhs(b, j, k);
                }
            }
        }
    }

    return result;
}


inline extent get_batch_shape(extent const &shape) {
    if (shape.size() < 3) throw NotEnoughDimensions(shape);
    std::size_t count = shape.size() - 2;

    extent batch_shape(count);
    std::copy_n(std::begin(shape), count, std::begin(batch_shape));
    return batch_shape;
}

inline extent calculate_batch_shape(extent const &shape, index_t d0, index_t d1) {
    auto new_shape = get_batch_shape(shape);
    new_shape.push_back(d0);
    new_shape.push_back(d1);
    return new_shape;
}

template <typename T, typename Device>
Tensor<T, Device> product(Tensor<T, Device> const &lhs,
                          Tensor<T, Device> const &rhs)
{
    auto lhs_dims = num_dims(lhs);
    auto rhs_dims = num_dims(rhs);

    // 5x3x6 * 5x
    if (lhs_dims > 2 || rhs_dims > 2) {
        auto lhs_copy = lhs;
        auto rhs_copy = rhs;

        // flatten batch dimension if either tensor is > 3 dims
        if (lhs_dims > 3) {
            lhs_copy = reshape(lhs_copy, {expand, lhs.shape()[lhs_dims-2], lhs.shape()[lhs_dims-1]});
        }

        if (rhs_dims > 3) {
            rhs_copy = reshape(rhs_copy, {expand, rhs.shape()[rhs_dims-2], rhs.shape()[rhs_dims-1]});
        }

        // if (lhs_dims == 2) {
        //     // add batch dimension
        // } else if (lhs_dims == 1) {
        //     // treat as batch matrix-vector multiplication
        // }

        // 2. either lhs or rhs > 3, in which case need to flatten
        auto result = batch_matrix_matrix_product(lhs_copy, rhs_copy);

        // for now, assume both lhs and rhs start off as the same shape
        // auto new_shape = get_batch_size(orig_lhs_shape);
        auto new_shape = calculate_batch_shape(lhs.shape(), result.shape()[1], result.shape()[2]);

        // reshape into original batch dims
        return reshape(result, new_shape);
    }

    if (lhs_dims == 1 && rhs_dims == 1) {
        return vector_vector_product(lhs, rhs);
    }

    if (rhs_dims == 1) {
        return matrix_vector_product(lhs, rhs);
    }

    if (lhs_dims == 1) {
        // TODO: follow pytorch convention and add a dimension
        // to lhs and make matrix_matrix?
        return matrix_vector_product(rhs, lhs);
    }

    // matrix output
    return matrix_matrix_product(lhs, rhs);
}

template <typename T, typename Device>
Tensor<T, Device> operator *(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return product(lhs, rhs);
}


template <typename T, typename Device>
Tensor<std::uint8_t, Device> operator ==(Tensor<T, Device> const &lhs,
                                         Tensor<T, Device> const &rhs)
{
    return apply<std::uint8_t>(lhs, rhs, [](T const &lop, T const &rop) {
        return lop == rop;
    });
}

template <typename T, typename Device>
Tensor<std::uint8_t, Device> operator !=(Tensor<T, Device> const &lhs,
                                         Tensor<T, Device> const &rhs)
{
    return !(lhs == rhs);
}

template <typename T, typename Device, typename F>
bool all(Tensor<T, Device> const &op, F fn) {
    for (auto const &index : op.indices()) {
        if (!fn(op(index))) return false;
    }

    return true;
}

template <typename T, typename Device, typename F>
bool any(Tensor<T, Device> const &op, F fn) {
    for (auto const &index : op.indices()) {
        if (fn(op(index))) return true;
    }

    return false;
}

template <typename T, typename Device>
bool equals(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return all(lhs == rhs, [](std::uint8_t v) { return v == true; });
}

#endif