# Tensor Library for C++

## Goal

To create a modern c++ tensor library for instructional purposes.

### Why c++?

The c++11 language is a very different than c++98. The soon to be finished c++20 language has continued in that vein, and has added a large set of features that make c++ a good choice for both math and ML applications.

### Why not use Python?

It's not possible to write the library entirely in Python, so some compiled language has to be used. So the question is what does Python provide that isn't already available in c++? Conventional wisdom is that Python is easy to use and c++ is difficult. And while to some extent that's true, modern c++ has simplified a lot of aspects of the language, to the point that if you're not writing a library, often the code itself is often close to what you would write in Python.

For example here is some simple code in Python:
```python
# create a 2d tensor
a = np.array([[1, 2, 3], [4, 5, 6])

# slice tensor
b = a[:, 1:]

# display new tensor
print("b = \n{}", b)
```

and the corresponding code for the tensor library:
```c++
//  create a 2d tensor
auto a = make_tensor({{1, 2, 3}, {4, 5, 6}});

// slice tensor
tensor b = a[{0, 1}][{1, 2}];

// display new tensor
fmt::print("b = \n{}\n", b);
```

Additionally, c++ has features that python doesn't have:
1. **Strong typing**: It's easy to know the type of every variable. For large projects this can radically decrease development time. Additionally, IDEs can help automatically detect errors for you.
2. **Compiled**: A compiled language allows you to catch many errors before you run. Especially in cases where you may train a model for long periods of time, it's better to know your code doesn't have errors.
3. **Simplicity**: If you need to develop new features, you will either have to write bridge code to Python, or use a library that automates that. Additionally, this makes it more difficult to both understand how the code works, as well as debug your code.