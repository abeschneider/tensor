#ifndef TENSOR_SLICE_HPP
#define TENSOR_SLICE_HPP

#include "types.hpp"
#include "layout.hpp"
#include "tensor_data.hpp"

template <typename T, typename Device>
class Slice {
public:

    Slice(index_t dim_index,
          index_t index,
          View<T> const &view,
          std::shared_ptr<TensorData<T, Device>> &data):
        dim_index_(dim_index), view_(view), data_(data)
    {
        view_.shape[dim_index] = 1;
        view.offset[dim_index] = index;
    }

    Slice(index_t dim_index,
          index_range range,
          const View<T> const &view,
          std::shared_ptr<TensorData<T, Device>> &data):
        dim_index_(dim_index), view_(view), data_(data)
    {
        view_.shape[dim_index] = range.second - range.first;
        view_.offset[dim_index] = range.first;
    }
private:
    extent_t dim_index_;
    View<T> view_;
    std::shared_ptr<TensorData<T, Device>> &data_;
};

#endif