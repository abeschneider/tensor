#ifndef INDEX_GENERATOR_HPP
#define INDEX_GENERATOR_HPP

#include <algorithm>
#include <numeric>
#include <range/v3/all.hpp>

#include "types.hpp"
#include "layout.hpp"

class index_generator: public ranges::view_facade<index_generator, ranges::finite> {
public:
    friend ranges::range_access;

    index_generator() = default;

    index_generator(extent const &shape, indices const &strides):
        shape_(shape),
        strides_(strides),
        index_(shape.size(), 0),
        count_(0),
        max_count_(::num_elements(shape_)) {}

    explicit index_generator(extent const &shape):
        index_generator(shape, make_strides(shape, make_row_major_order(shape.size()))) {}

    explicit index_generator(Layout const &layout):
        index_generator(layout.shape, layout.strides) {}

    explicit index_generator(View const &view):
        index_generator(view.shape, make_strides(view.shape, view.order)) {}
private:
    void update_index(index_t value) {
        auto write = std::begin(index_);
        auto extent = std::begin(shape_);

        for (auto const &stride : strides_) {
            *write++ = index_t(value / stride) % *extent++;
        }
    }
public:
    void next() {
        ++count_;
        update_index(count_);
    }

    bool equal(ranges::default_sentinel_t) const {
        return count_ == max_count_;
    }

    const extent &read() const { return index_; }
private:
    extent shape_;
    indices strides_;
    indices index_;

    index_t count_;
    index_t max_count_;
};

#endif