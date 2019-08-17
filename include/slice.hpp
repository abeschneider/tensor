#ifndef SLICE_HPP
#define SLICE_HPP

#include <utility>

#include <range/v3/all.hpp>

#include "types.hpp"
#include "layout.hpp"

using index_range_t = std::pair<std::size_t, std::size_t>;

template <typename T, typename D>
class Slice {
public:
    using NumericType = T;
    using Device = D;

    Slice(index_t dim_index,
          index_t index,
          View const &view,
          std::shared_ptr<Storage<T, Device>> storage):
            dim_index_(dim_index), view_(view), storage_(storage)
    {
        // view_.shape(dim_index_) = 1;
        view_.shape[dim_index] = 1;
        view_.offset[dim_index_] = index + view.offset[dim_index_];
    }

    Slice(index_t dim_index,
          index_range_t index_range,
          View const &view,
          std::shared_ptr<Storage<T, Device>> storage):
            dim_index_(dim_index), view_(view), storage_(storage)
    {
        // view_.shape(dim_index_) = index_range.second - index_range.first;
        view_.shape[dim_index] = index_range.second - index_range.first;
        view_.offset[dim_index_] = index_range.first;
    }

    Slice<T, Device> operator [](index_t index) const {
        return Slice(dim_index_+1, index, view_, storage_);
    }

    Slice<T, Device> operator [](index_range_t index_range) const {
        return Slice(dim_index_+1, index_range, view_, storage_);
    }

    View const &view() const { return view_; }
    View &view() { return view_; }

    extent const &shape() const { return view_.shape; }
    index_t shape(index_t dim) const { return view_.shape[dim]; }

    Storage<T, Device> const &storage() const { return *storage_; }
    std::shared_ptr<Storage<T, Device>> storage_ptr() const { return storage_; }

    // TODO: rename dim_index, confusing otherwise
    index_t index() const { return dim_index_; }
    index_t num_dims() const { return view_.shape.size(); }
private:
    index_t dim_index_;
    View view_;
    std::shared_ptr<Storage<T, Device>> storage_;
};

// template <typename Sliceable>
// class SliceGenerator: public ranges::view_facade<SliceGenerator<Sliceable>, ranges::finite> {
// public:
//     friend ranges::range_access;

//     using NumericType = typename Sliceable::NumericType;
//     using Device = typename Sliceable::Device;

//     SliceGenerator(): sliceable_(nullptr) {}

//     explicit SliceGenerator(Sliceable &sliceable):
//         sliceable_(&sliceable), index_(0) {}

//     void next() {
//         ++index_;
//     }

//     Slice<NumericType, Device> read() const {
//         return (*sliceable_)[index_];
//     }

//     bool equal(ranges::default_sentinel_t) const {
//         return index_ == sliceable_->shape(sliceable_->index());
//     }
// private:
//     Sliceable *sliceable_;
//     index_t index_;
// };

#endif