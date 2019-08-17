#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

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

// TODO: make this either respect order, or take order f strides
template <typename T, typename Device>
std::optional<Tensor<T, Device>> reshape(
    Tensor<T, Device> const &tensor,
    extent const &shape)
{
    std::optional<Tensor<T, Device>> result;

    // we can only reshape if the number of elements is conserved
    if (num_elements(shape) == num_elements(tensor.shape())) {
        auto order = make_row_major_order(shape.size());
        auto strides = make_strides(shape, order);
        auto offset = make_offset(shape.size());
        result = tensor.view({shape, offset, order, strides});
    }

    return result;
}

template <typename T, typename Device>
bool is_broadcastable_to(Tensor<T, Device> const &tensor,
                         extent const &shape)
{
    if (shape.size() <= tensor.shape().size()) return false;

    auto shape_it = shape.rbegin();
    auto tensor_shape_it = tensor.shape().rbegin();
    for (; tensor_shape_it != tensor.shape().rend(); ++shape_it, ++tensor_shape_it) {
        if (*shape_it != *tensor_shape_it && *tensor_shape_it != 1) return false;
    }

    return true;
}

template <typename T, typename Device>
std::optional<Tensor<T, Device>> broadcast_to(
    Tensor<T, Device> const &tensor,
    extent const &shape)
{
    std::optional<Tensor<T, Device>> result;

    extent strides;
    auto offset = make_offset(shape.size());
    auto order = make_row_major_order(shape.size());

    if (is_broadcastable_to(tensor, shape)) {
        for (int i = shape.size()-1; i >= 0; i--) {
            if (i < tensor.shape().size() && tensor.shape()[i] > 1) {
                strides.push_back(tensor.view().strides[i]);
            } else {
                strides.push_back(0);
            }
        }

        result = tensor.view({shape, offset, order, strides});
    }

    return result;
}

#endif