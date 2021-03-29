# Input Types

Stheno enables the user to handle any type of input domain they wish and provides a common API that users must implement when considering a new way of representing input data to ensure that the package knows how to handle them appropriately.

This interface has now been adopted throughout the JuliaGaussianProcesses ecosystem.

Discussed below is this interface's core assumptions, a detailed account of a couple of important concrete input types.

## The Central Assumption

The central assumption made in all user-facing and internal functions is this: **when a collection of inputs are required, they subtype `AbstractVector`**. For example, `x isa AbstractVector` when indexing into a GP:
```julia
f(x, σ²)
```
or computing the covariance matrix associated with a kernel:
```julia
kernelmatrix(SqExponentialKernel(), x)
```
When computing the cross-covariance matrix between two GPs
```julia
cov(f, g, x_f, x_g)
```
`x_f` and `x_g` must be `AbstractVector`s. _All other operations involving collections of inputs have the same restrictions_. Always `AbstractVector`s.

For example, this means that when handling multi-dimensional inputs stored in a `Matrix` they must be wrapped so that the package knows to treat them as a vector — more on this in below in _D-dimensional Euclidean Spaces_.



## 1-Dimensional Euclidean Space

When constructing a GP whose domain is the real-line, for example when using a GP to solve some kind of time-series problem, it is sufficient to work with `Vector{<:Real}`s of inputs. As such, the following is completely valid:
```julia
using Stheno: GPC
f = GP(SqExponentialKernel(), GPC())
x = randn(10)
f(x)
```
It is also possible to work with other `AbstractArray`s that represent a vector of 1-dimensional points, e.g.
```julia
x = range(-5.0, 5.0; length=100)
f(x)
```



## D-Dimensional Euclidean Space

Many applications of interest involve more than a single input-dimension, such as spatio-temporal modeling or Machine Learning tasks. For these cases, we provide `ColVecs <: AbstractVector`.
```julia
X_data = randn(5, 100)
X = ColVecs(X_data)
```
tells Stheno that it should treat each column of `X_data` as a vector-valued input. Phrased differently, `X` is an `AbstractVector{T}` where `T <: Vector{<:Real}`, which stores its elements in memory as a dense matrix. This approach has the advantage of making it completely explicit how Stheno treats a matrix of data, and also simplifies quite a bit of the internal machinery, as all vectors of inputs can be assumed to be a subtype of `AbstractVector`.

If, on the other hand, each _row_ of `X_data` corresponds to a vector-valued input, use `RowVecs(X_data)`.



### Structure in D-Dimensional Euclidean Space

Consider a rectilinear grid of points in D-dimensional Euclidean space. Such grids of points can be represented in a more memory-efficient manner than can arbitrarily locate sets of points. Moreover, this structure can be exploited to accelerate inference for certain types of problems dramatically. Other such examples exist e.g., uniform grids in N-dimensions, and can be exploited to more efficiently represent input data and to accelerate inference.

Work to exploit these kinds of structures is on-going at the time of writing and will be documented before merging.



## GPPPInput and BlockData

A `GPPPInput` is a special kind of `AbstractVector`, specifically designed for `GPPP`s.
Given a `GPPP`:
```julia
f = @gppp let
    f1 = GP(SEKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end
```
a `GPPPInput` like
```julia
x_ = randn(5)
x = GPPPInput(:f3, x_)
```
applied to `f`
```julia
fx = f(x, 0.1)
```
constructs a `FiniteGP` comprising `f3` at `x_`.
`GPPPInput(:f2, x_)` and `GPPPInput(:f1, x_)` are the analogues for `f1` and `f2`.

If you wish to refer to multiple processes in `f` at the same time, use a `BlockData`.
For example
```julia
x = BlockData(GPPPInput(:f2, x_), GPPPInput(:f3, x_))
```
would pull out a 10-dimensional `FiniteGP`, the first 5 dimensions being `f2` at `x_`, the last 5 dimensions being `f3` at `x_`.
