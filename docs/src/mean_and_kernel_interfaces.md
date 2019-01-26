# Interfaces


## AbstractDataSets

There are multiple formats in which one could provide a "data set". For example, is might be a vector of numbers where the interpretation is that each element is a data point. Conversely, one might provide a data set in the form of a matrix, in which case the interpretation is not unambiguous -- should each element be considered a data point? This is not usually the interpretation the people have in mind; typically a matrix is provided because one has vector-valued observations, and each column or row corresponds to a data point. So there are essentially not-unreasonable obvious interpretations of a data set provided as a matrix, and different people have different ideas about what is "right".

To make matters even more ambiguous, what if I want to provide a vector (whose length is an even number) where each pair of elements corresponds to an observation? This might seem like a slightly strange scheme, and I admit that I can't think of a particular use-case off the top of my head, but it doesn't seem completely unreasonable to me that one might want to do such a thing.

Moreover, the types of computations that one wishes to perform will likely vary depending upon the type of data set one is provided with. For example, the particular way in which we will compute the covariance matrix of the Exponentiated Quadratic kernel will depend on the dimensionality of the input data, and how it is stored in memory.

The above motivates abstracting away the implementation details of any particular data set. We propose to do this is in the usual way using the type system. A traits-based system would probably also work, but we don't consider that for now.

We propose the `AbstractDataSet` type, which should be thought of conceptually as an ordered list of "data points", where a data point is whatever you define it to be (common examples being `Real`s, and `Vector{<:Real}`s). An `AbstractDataSet` `D` should adhere to the following interface:

| Required methods | | Brief description |
|:--------------------- |:---------------------- |:-------------------------------- |
| `size(D)`             |  | The size of `D`. Should be a 1-Tuple. |
| `getindex(D, n::Int)` |  | Get the `n`th data point. |
| `getindex(D, n::AbstractVector{Int})` |  | Get the data points at each index in `n`. |
| `eachindex(D)`         |  | All the indices of `D`.       |
| **Important optional methods** | **Default definition** | **Brief description**   |

At the current time, `AbstractDataSet` subtypes `AbstractVector`, as we wish `map` and `broadcast` to work properly with it. As of 0.7 / 1.0, this will be unnecessary.

### Built-In `AbstractDataSet`s

#### VectorData
A thin wrapper around a vector of data points of arbitrary type.

#### MatrixData
A thin wrapper around a matrix in which each column should be a data point.

#### BlockData
Defined in terms of a vector of `AbstractDataSet`s.



### `map` and `pairwise` with `AbstractDataSet`s

From `Stheno`'s perspective, the two most important operations involving `AbstractDataSet`s are `map` and `pairwise`. Mapping a unary, scalar-valued, function `f` over an `AbstractDataSet` `D` of length `N` should return an `AbstractVector` of length `N`, whilst applying `pairwise` to a binary, scalar-valued, function `f` in conjunction with two `AbstractDataSet`s `D1` and `D2` of lengths `N1` and `N2` respectively should return an `AbstractMatrix` of size `N1 x N2`. Note that we require that the return types of these operations are specified abstractly. This is crucial since certain return types might have special structure, the most crucial of which being block structure.

| Required methods | Description |
|:--------------------- |:-------------------------------- |
| `map(f, D)`           | The usual definition of `map` for unary `f` and iterable `D`. |
| `map(k, D1, D2)`        | Usual definition of `map` for binary `k` and iterables `D1`, `D2`. |
| `pairwise(k, D1, D2)` |`N1` x `N2` `AbstractMatrix` `K` where `K[p, q] = f(D1[p], D2[q])`.|
| `pairwise(k, D)`      | Shorthand for `pairwiise(f, D, D)`. |





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







## Abstract Gaussian Process interface

For `f, g <: AbstractGaussianProcess`, we require the following methods:

| Required methods | Brief description |
|:--------------------- |:-------------------------------- |
| `mean(f)` | Representation of the mean function of a GP |
| `kernel(f)` | Representation of the kernel of a GP |
| `kernel(f, g)` | Representation of the cross-kernel between `f` and `g` |
| **Important optional methods** | **Brief description** |
| `f == g` | Equality. Defaults to checking mean and kernel are equal |
| `length(f)` | Defaults to length of `length(mean(f))` |
| `eachindex(f)` | Defaults to `eachindex(mean(f))` |
| `mean(f)` | Vector representation of `mean(f)`. Defaults to `AbstractVector(mean(f))`|
| `cov(f)` | Covariance matrix of `f`. Defaults to `AbstractMatrix(kernel(f))` |
| `cov(f, g)` | Cross-covariance matrix between `f` and `g`. Defaults to `AbstractMatrix(kernel(f, g))` |
| `marginal_cov(f)` | Equivalent to `diag(cov(f))`, but fast |

Note that `eachindex`, `mean`, `cov`, `cov`, and `marginal_cov` should all fail if the dimensionality of `f` (or `g` if applicable) is not finite.
