# Input Types

Stheno enables the user to handle any type of input domain they wish, and provides a common API that must be implemented when considering a new way of representing input data so as to ensure that the package knows how to handle them appropriately.

Discussed below is this interface's core assumptions, a detailed account of a couple of important concrete input types. Additionally, a worked-example of a new input type is provided.



## The Central Assumption

The central assumption made in all user-facing and internal functions is this: **when a collection of inputs are required, they will subtype `AbstractVector`**. For example, `x isa AbstractVector` when indexing into a GP:
```julia
f(x, σ²)
```
or computing the covariance matrix assocated with a kernel:
```julia
pw(eq(), x)
```
When computing the cross-covariance matrix between two GPs
```julia
cov(f, g, x_f, x_g)
```
`x_f` and `x_g` must be `AbstractVector`s. _All other operations involving collections of inputs has the same restrictions_. Always `AbstractVector`s.

For example, this means that when handling multi-dimensional inputs stored in a `Matrix` they must be wrapped so that the package knows to treat them as a vector. More on this in below in _D-dimensional Euclidean Spaces_.



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

Many applications of interest involve more than a single input-dimension, such as spatio-temporal modeling or Machine Learning tasks. For these cases we provide `ColVecs <: AbstractVector`.
```julia
X_data = randn(5, 100)
X = ColVecs(X_data)
```
tells Stheno that it should treat each column of `X_data` as a vector-valued input. Phrased differently, `X` is an `AbstractVector{T}` where `T <: Vector{<:Real}` whose elements are stored in memory as a dense matrix. This approach has the advantage of making it completely explicit how Stheno will treat a matrix of data, and also simplifies quite a bit of the internal machinery, as all vectors of inputs can actually be assumed to be a subtype of `AbstractVector`.

Future plans include a `RowVecs` type which would instead treat each row of `X_data` as a vector-valued input. If you would like this feature, please raise an issue or a PR to let us know there's demand for it. The worked example below actually makes some headway on this, so it provides an excellent starting point for a PR!



### Structure in D-Dimensional Euclidean Space

Consider a rectilinear grid of points in D-dimensional Euclidean space. Such grids of points can be represented in a more memory-efficient manner than can arbitrarily located sets of points. Moreover, this structure can be exploited to dramatically accelerate inference for certain types of problems. Other such examples exist e.g. uniform grids in N-dimensions, and can be exploited to more efficiently represent input data and to accelerate inference.

Work to exploit these kinds of structure is on-going at the time of writing, and will be documented before merging.


## Worked Example

As discussed, `ColVecs` is already supported for inputs in D-dimensional Euclidean space, where a collection of inputs is stored in a `Matrix` and each **column** is an input. The following example presents `RowVecs`, which is also made for representing collections inputs residing in D-dimensional Euclidean space but the interpretation is different: each **row** of the matrix corresponds to an input.

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
This structure will now print nicely, and pass some consistency checks, but none of the base `Kernel`s in the package know how to treat it. This means that, for example, new `pw` and `ew` methods that are specialised to `RowVec` must be added:
```julia
import Stheno: pw, EQ
using Distances: SqEuclidean
pw(k::EQ, x::RowVecs, x′::RowVecs) = exp.(.-pw(SqEuclidean(), x.X, x′.X; dims=1) ./ 2)
# insert implementations for the unary pw and the two ew methods
```
Unfortunately, this really _does_ mean that every `Kernel` needs 4 new methods to handle a new data structure. While the implementation will usually be straightforward, this is somewhat more laborious than is ideal.

There are, however, a couple of tricks that can be employed to avoid this task. If it is the case that a particular data structure can be easily **converted** into e.g. a `ColVecs` object, then one can simply compose the GP of interest with a function doing this conversion:
```julia
# It needs to be possible to efficiently broadcast the composition operator
# over the `RowVecs` data type for the new `to_colvecs` function.
to_colvecs(x) = error("Shouldn't ever hit this")
Base.broadcasted(::typeof(to_colvecs), x::RowVecs) = ColVecs(x.X')

using Stheno: GPC
f = GP(eq(), GPC())
g = f ∘ to_colvecs # \circ

x = RowVecs(randn(10, 3))
rand(g(x))
```
