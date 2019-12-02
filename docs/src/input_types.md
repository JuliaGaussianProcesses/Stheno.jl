# Input Types

Stheno enables the user to handle any type of input domain they wish and provides a common API that users must implement when considering a new way of representing input data to ensure that the package knows how to handle them appropriately.

Discussed below is this interface's core assumptions, a detailed account of a couple of important concrete input types. Additionally, we provide a worked-example of a new input type.

## The Central Assumption

The central assumption made in all user-facing and internal functions is this: **when a collection of inputs are required, they subtype `AbstractVector`**. For example, `x isa AbstractVector` when indexing into a GP:
```julia
f(x, σ²)
```
or computing the covariance matrix associated with a kernel:
```julia
pw(eq(), x)
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
f = GP(eq(), GPC())
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

Future plans include a `RowVecs` type, which would instead treat each row of `X_data` as a vector-valued input. If you would like this feature, please raise an issue or a PR to let us know there's a demand for it. The worked example below actually makes some headway on this, so it provides an excellent starting point for a PR!



### Structure in D-Dimensional Euclidean Space

Consider a rectilinear grid of points in D-dimensional Euclidean space. Such grids of points can be represented in a more memory-efficient manner than can arbitrarily locate sets of points. Moreover, this structure can be exploited to accelerate inference for certain types of problems dramatically. Other such examples exist e.g., uniform grids in N-dimensions, and can be exploited to more efficiently represent input data and to accelerate inference.

Work to exploit these kinds of structures is on-going at the time of writing and will be documented before merging.


## Worked Example

As discussed, `ColVecs` is already supported for inputs in D-dimensional Euclidean space, where `Matrix` stores a collection of inputs, and each **column** is an input. The following example presents `RowVecs`, which represents collections inputs residing in D-dimensional Euclidean space, but the interpretation is different: each **row** of the matrix corresponds to an input.

Firstly, the new data structure is specified:
```julia
struct RowVecs{T<:Real} <: AbstractVector{Vector{T}}
    X::Matrix{T}
end
```
Observe that it subtypes `AbstractVector`, and each element is a `Vector{T<:Real}`. It has a single field `X`, which is a matrix of elements. It is necessary to implement some parts of the [AbstractArray interface](https://docs.julialang.org/en/v1/manual/interfaces/index.html#man-interface-array-1) to ensure printing and various consistency checks inside the package work as intended:
```julia
Base.length(x::RowVecs) = size(x.X, 1)
Base.size(x::RowVecs) = (length(x),)
Base.getindex(x::RowVecs, n::Int) = x.X[n, :]
```
This structure prints nicely and pass some consistency checks, but none of the base `Kernel`s in the package know how to treat it. This means that, for example, new `pw` and `ew` methods that are specialised to `RowVec` must be added:
```julia
import Stheno: pw, EQ
using Distances: SqEuclidean
pw(k::EQ, x::RowVecs, x′::RowVecs) = exp.(.-pw(SqEuclidean(), x.X, x′.X; dims=1) ./ 2)
# insert implementations for the unary pw and the two ew methods
```
In the worst case, this means that every `Kernel` needs four new methods to handle a new data structure. Fortunately, this case isn't typical. Most of the `Kernel`s in `src/kernels.jl` are implemented in terms of the `SqEuclidean` and `Euclidean` distances, and are generically typed. As such it is sufficient to add new `ew` and `pw` methods involving those types, without the need to re-implement those methods for kernels. See [src/util/distances.jl](https://github.com/willtebbutt/Stheno.jl/blob/master/src/util/distances.jl) for examples.
