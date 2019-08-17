#ifndef FORMAT_HPP
#define FORMAT_HPP

#include <fmt/format.h>

template <typename OutputIterator>
void repeat(OutputIterator write, const char ch, std::size_t count) {
    for (std::size_t i = 0; i < count; i++) {
        fmt::format_to(write, "{}", ch);
    }
}

// displays the inner most dimension as a row of values
template <typename OutputIterator, typename Sliceable>
void format_inner(OutputIterator write, Sliceable const &slice) {
    index_t last_dim_size = slice.shape(slice.num_dims()-1);

    fmt::format_to(write, "[");
    for (index_t i = 0; i < last_dim_size; i++) {
        auto offset = calculate_offset(slice.view(), i);

        fmt::format_to(write, "{:3}", slice.storage()[offset]);

        if (i < last_dim_size-1) {
            fmt::format_to(write, ", ");
        }
    }

    fmt::format_to(write, "]");
}

template <typename OutputIterator, typename Sliceable>
void format_outer(OutputIterator write, Sliceable const &slice, index_t index) {
    for (index_t i = 0; i < slice.shape(index); i++) {
        if (i == 0) {
            fmt::format_to(write, "[");
        } else {
            repeat(write, ' ', index+1);
        }

        auto next_slice = slice[i];
        if (next_slice.index() == next_slice.num_dims()-2) {
            format_inner(write, next_slice);
        } else {
            format_outer(write, next_slice, next_slice.index()+1);
        }

        if (i == slice.shape(index)-1) {
            fmt::format_to(write, "]");
        } else {
            fmt::format_to(write, ",");
            repeat(write, '\n', (slice.num_dims() - index)-1);
        }
    }
}

template <typename OutputIterator, typename T, typename Device>
void format_tensor(OutputIterator write, Tensor<T, Device> const &tensor) {
    if (tensor.num_dims() > 1) {
        format_outer(write, tensor, 0);
    } else {
        format_inner(write, tensor);
    }
}

namespace fmt {

template <typename T, typename Device>
struct formatter<Tensor<T, Device>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Tensor<T, Device> &t, FormatContext &ctx) {
        format_tensor(ctx.out(), t);
        return ctx.out();
    }
};

} // namespace fmt


template <typename T, typename Device>
std::ostream& operator<<(std::ostream &os, Tensor<T, Device> const &tensor) {
    fmt::print(os, "{}", tensor);
    return os;
}

#endif