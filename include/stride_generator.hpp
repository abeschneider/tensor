#ifndef STRIDE_GENERATOR_HPP
#define STRIDE_GENERATOR_HPP

#include <range/v3/all.hpp>

#include "types.hpp"

class ordered_shape_view: public ranges::view_facade<ordered_shape_view, ranges::finite> {
public:
    ordered_shape_view() = default;
    ordered_shape_view(extent const &shape, indices const &order):
        shape_(&shape), order_(&order), index_(0) {}

    void next() { ++index_; }
    const index_t &read() const { return (*shape_)[(*order_)[index_]]; }
    std::size_t size() const { return shape_->size(); }

    bool equal(ranges::default_sentinel_t) const {
        return index_ >= shape_->size();
    }
private:
    extent const *shape_;
    indices const *order_;
    index_t index_;
};


class stride_generator: public ranges::view_facade<stride_generator, ranges::finite> {
public:
    friend ranges::range_access;

    stride_generator() = default;
    explicit stride_generator(ordered_shape_view const &shape):
        shape_(shape), stride_(1), count_(0) {}

    void next() {
        stride_ *= shape_.read();
        shape_.next();
        ++count_;
    }

    const index_t &read() const { return stride_; }

    bool equal(ranges::default_sentinel_t) const {
        return count_ >= shape_.size();
    }

private:
    ordered_shape_view shape_;
    index_t stride_;
    index_t count_;
};

inline indices make_strides(extent const &shape, const indices &order) {
    indices result(shape.size());

    ordered_shape_view shape_view(shape, order);
    stride_generator strides(shape_view);

    std::size_t i = 0;
    for (auto stride : strides) {
        result[order[i++]] = stride;
    }

    return result;
}

inline indices make_row_major_order(std::size_t num_dims) {
    indices result = ranges::view::iota(0) | ranges::view::take(num_dims) | ranges::view::reverse;
    return result;
}

inline indices make_col_major_order(std::size_t num_dims) {
    indices result = ranges::view::iota(0) | ranges::view::take(num_dims);
    return result;
}

#endif
