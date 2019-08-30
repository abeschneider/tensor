#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <vector>

#include "types.hpp"

// namespace tensor {

struct CPU {};
struct CPU_BLAS {};
struct GPU_CUDA {};

/**
 * @brief Storage type for `Tensor`s.
 *
 * @tparam T Element type of storage
 * @tparam Device Device storage lives on
 */
template <typename T, typename Device>
struct Storage {};

/**
 * @brief CPU specialization of `Storage`
 *
 * @tparam T Element type of storage
 */
template <typename T>
struct Storage<T, CPU> {
    using element_type = T;
    using storage_type = std::vector<element_type>;
    using iterator = typename storage_type::iterator;

    Storage(std::size_t size): data(size, 0) {}

    T &operator [](index_t i) { return data[i]; }
    const T &operator [](index_t i) const { return data[i]; }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }

    auto cbegin() const { return data.cbegin(); }
    auto cend() const { return data.cend(); }

    std::size_t size() const { return data.size(); }

    std::vector<T> data;
};

#endif