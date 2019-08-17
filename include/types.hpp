#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <numeric>

using offset_t = std::size_t;
using index_t = std::size_t;

// make extent_t
using extent = std::vector<index_t>;

// make indices_t
using indices = std::vector<index_t>;
using index_range = std::pair<index_t, index_t>;

inline std::size_t num_elements(const extent &shape) {
    return std::accumulate(
        std::begin(shape), std::end(shape), 1, std::multiplies<>());
}

#endif