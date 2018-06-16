# Interfaces


## Means, Kernels, and CrossKernels

`MeanFunction`s, `Kernel`s, and `CrossKernel`s form the core of Stheno. All of the Gaussian process and affine transform functionality is built on top of / defined in terms of these basic components. Consequently, it is especially important that they be defined in terms of a consistent interface, and thoroughly tested.

We present the interfaces for each of these components. Furthermore, we discuss the features and limitations of the built-in testing functionality for each of the components, explaining what problems you can expect the tests to identify for you, and conversely what types of tests cannot be automated.


## `MeanFunction`

For any `μ isa MeanFunction`, we have the following:

| Required methods | | Brief description |
|:--------------------- |:---------------------- |:-------------------------------- |
| `μ(x)`         |  | Evaluate `μ` at `x`.        |
| `eachindex(μ)` |  | Index set of domain of `μ`. Should error if `length(μ)` is infinite.
| **Important optional methods** | **Default definition** | **Brief description**                                                                 |
| `unary_obswise(μ, X)` | `unary_obswise_fallback`  | `μ(x)` for each observation `x` in `X` |
| `length(μ)`           | `∞`                       | The size of the domain of `μ` |

Required testing is as follows:

| Required methods | Automatic Tests | Tests to Write |
|:------------------------ |:---------------------- |:-------------------------------- |
| `μ(x)`         | Is not `Nothing` | Correctness |
| `eachindex(μ)` | No testing | Correctness |
| **Important optional methods** | **Automatic Tests** | **Tests to Write** |
| `unary_obswise(μ, X)` | Is consistent with `unary_obswise_fallback` | None |
| `length(μ)`           | No testing | Correctness |


## `Kernel`

