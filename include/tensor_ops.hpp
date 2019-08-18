#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdexcept>
#include <fmt/format.h>

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

/**
 * \brief Sets all the elements of a tensor to the given value
 * \param t The tensor to operate on
 * \param value The value to use
 */
template <typename T, typename Device>
void fill(Tensor<T, Device> &t, T const &value) {
    for (auto const &index : t.indices()) {
        t(index) = value;
    }
}

template <typename T, typename Device=CPU>
Tensor<T, Device> zeros(extent const &shape) {
    Tensor<T, Device> result(shape);
    fill(result, 0);
    return result;
}

template <typename T, typename Device=CPU>
Tensor<T, Device> ones(extent const &shape) {
    Tensor<T, Device> result(shape);
    fill(result, 0);
    return result;
}

/**
 * \brief Re-orders dimensions of Tensor
 * \param tensor The Tensor to re-order
 * \param indices The new order of dimensions
 * \return The re-ordered tensor.
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

// TODO: get rid of optional, and make invalid Tensor
// TODO: make this either respect order, or take order f strides
template <typename T, typename Device>
Tensor<T, Device> reshape(Tensor<T, Device> const &tensor, extent const &shape) {
    // we can only reshape if the number of elements is conserved
    if (num_elements(shape) != num_elements(tensor.shape())) {
        throw MismatchedNumberOfElements(num_elements(shape),
                                         num_elements(tensor.shape()));
    }

    auto order = make_row_major_order(shape.size());
    auto strides = make_strides(shape, order);
    auto offset = make_offset(shape.size());
    return tensor.view({shape, offset, order, strides});
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
    extent strides;
    auto offset = make_offset(shape.size());
    auto order = make_row_major_order(shape.size());

    for (int i = shape.size()-1; i >= 0; i--) {
        if (i < tensor.shape().size() && tensor.shape()[i] > 1) {
            strides.push_back(tensor.view().strides[i]);
        } else {
            strides.push_back(0);
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
    if (lhs.shape() != rhs.shape()) { //return Tensor<RT, Device>();
        throw MismatchedDimensions(lhs.shape(), rhs.shape());
    }

    Tensor<RT, Device> result(lhs.shape());

    // can optimize if both lhs and rhs are contiguous
    for (auto const &index : lhs.indices()) {
        result(index) = fn(lhs(index), rhs(index));
    }

    return result;
}

template <typename T, typename Device, typename F>
void apply_inplace(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs, F fn) {
    if (lhs.shape() != rhs.shape()) {
        throw MismatchedDimensions(lhs.shape(), rhs.shape());
    }

    // can optimize if both lhs and rhs are contiguous
    for (auto const &index : lhs.indices()) {
        lhs(index) = fn(lhs(index), rhs(index));
    }
}

template <typename T, typename Device>
Tensor<T, Device> operator +(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop + rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator +=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    apply_inplace(lhs, rhs, [](T const &lop, T const &rop) { return lop + rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> operator -(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop - rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator -=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    apply_inplace(lhs, rhs, [](T const &lop, T const &rop) { return lop - rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> operator *(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop * rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator *=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    apply_inplace(lhs, rhs, [](T const &lop, T const &rop) { return lop * rop; });
    return lhs;
}

template <typename T, typename Device>
Tensor<T, Device> operator /(Tensor<T, Device> const &lhs, Tensor<T, Device> const &rhs) {
    return apply<T>(lhs, rhs, [](T const &lop, T const &rop) { return lop / rop; });
}

template <typename T, typename Device>
Tensor<T, Device> &operator /=(Tensor<T, Device> &lhs, Tensor<T, Device> const &rhs) {
    apply_inplace(lhs, rhs, [](T const &lop, T const &rop) { return lop / rop; });
    return lhs;
}

// move to detail
template <typename T, typename Device>
Tensor<T, Device> vector_vector_dot(Tensor<T, Device> const &lhs,
                                    Tensor<T, Device> const &rhs)
{
    if (num_elements(lhs) != num_elements(rhs)) {
        throw MismatchedNumberOfElements(num_elements(lhs), num_elements(rhs));
    }

    T result = 0.0;
    for (index_t i = 0; i < num_elements(lhs); i++) {
        result += lhs(i)*rhs(i);
    }

    return tensor({result});
}

template <typename T, typename Device>
Tensor<T, Device> matrix_vector_dot(Tensor<T, Device> const &lhs,
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

template <typename T, typename Device>
Tensor<T, Device> matrix_matrix_dot(Tensor<T, Device> const &lhs,
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
Tensor<T, Device> dot(Tensor<T, Device> const &lhs,
                      Tensor<T, Device> const &rhs)
{
    auto lhs_dims = num_dims(lhs);
    auto rhs_dims = num_dims(rhs);

    if (lhs_dims > 2 || rhs_dims > 2) {
        // currently this is unsupported
        // return Tensor<T, Device>();
        throw TensorError("Unsupported");
    }

    if (lhs_dims == 1 && rhs_dims == 1) {
        return vector_vector_dot(lhs, rhs);
    }

    if (rhs_dims == 1) {
        return matrix_vector_dot(lhs, rhs);
    }

    if (lhs_dims == 1) {
        return matrix_vector_dot(rhs, lhs);
    }

    // matrix output
    return matrix_matrix_dot(lhs, rhs);
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