#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include "types.hpp"
#include "stride_generator.hpp"

struct Layout {
    Layout() = default;

    Layout(Layout const &layout):
        shape(layout.shape), strides(layout.strides) {}

    // make sure we don't call this anywhere
    // Layout(extent const &shape, indices const &order):
    //     shape(shape), strides(make_strides(shape, order)) {}

    Layout(extent const &shape, indices const &strides):
        shape(shape), strides(strides) {}

    std::size_t num_elements() const { return ::num_elements(shape); }
    std::size_t size() const { return shape.size(); }

    offset_t get_offset(index_t dim, index_t index) const {
        return index*strides[dim];
    }

    extent shape;
    indices strides;
};

inline indices make_offset(std::size_t dims) {
    return indices(dims, 0);
}

inline indices increasing_order(std::size_t num_dims) {
    indices result = ranges::view::iota(0) | ranges::view::take(num_dims);
    return result;
}

struct View: public Layout {
    View(extent const &shape,
         indices const &offset,
         indices const &order,
         indices const &strides):
        Layout(shape, strides),
        offset(offset),
        order(order) {}

    View(Layout const &layout,
         extent const &shape,
         indices const &offset,
         indices const &order):
        Layout(shape, layout.strides),
        offset(offset),
        order(order) {}

    View(Layout const &layout,
         extent const &shape,
         indices const &offset):
        Layout(shape, layout.strides),
        offset(offset),
        order(increasing_order(shape.size())) {}

    View(Layout const &layout,
         extent const &shape):
        View(layout, shape, make_offset(shape.size()),
             increasing_order(shape.size())) {}

    explicit View(Layout const &layout):
        View(layout, layout.shape, make_offset(layout.shape.size()),
             increasing_order(layout.shape.size())) {}

    explicit View(View const &view):
        Layout(view), offset(view.offset), order(view.order) {}

    offset_t get_offset(index_t dim, index_t index) const {
        return (index+offset[dim])*strides[dim];
    }

    indices offset;
    indices order;
};

template <typename LayoutType, typename ContainerType>
offset_t calculate_offset(const LayoutType &layout, ContainerType const &index) {
    offset_t offset = 0;

    for (std::size_t d = 0; d < index.size(); d++) {
        offset += layout.get_offset(d, index[d]);
    }

    return offset;
}

// Q: keep this? provide a calculate_offset that ignores singleton and use that instead?
// or have both?
// TODO: provide check that all but one dimension is a singleton
template <typename LayoutType>
offset_t calculate_offset(const LayoutType &layout, index_t index) {
    offset_t offset = 0;

    for (index_t i = 0; i < layout.size(); i++) {
        if (layout.shape[i] == 1) {
            offset += layout.get_offset(i, 0);
        } else {
            offset += layout.get_offset(i, index);
        }
    }

    return offset;
}

#endif