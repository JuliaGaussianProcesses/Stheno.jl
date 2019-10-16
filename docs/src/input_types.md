# Input Types

Stheno enables the user to handle any type of input domain they wish, and provided a common API that must be implemented when considering a new type of domain. Probably the most common domains are the `Real`s and `Vector{<:Real}`, but it is possible to define GPs on strings, graphs, and other non-Euclidean spaces.

Stheno's core assumption is that the user will provide an `AbstractVector{T}` `x`, where `T` is the element type of `x`, whenever indexing. i.e. writing something of the form
```julia
f(x)
f(x, S)
```
This is different to many GP frameworks, that allow users to provide e.g. matrices of inputs to represent a vector of vector-inputs. It is explained below why we do _not_ opt for this choice.


## Univariate Domains

When constructing a GP whose domain is the real-line, for example when using a GP to solve some kind of time-series problem, it is sufficient to work with `Vector{<:Real}`s of inputs. For example...


## Euclidean Space

Many applications of interest involve more than a single input-dimension, such as spatio-temporal modeling or Machine Learning tasks. For these cases we provide `ColVecs <: AbstractVector`.
```julia
X_data = randn(5, 100)
X = ColVecs(X_data)
```
tells Stheno that it should treat each column of `X_data` as a vector-valued input. Phrased differently, `X` is an `AbstractVector{T}` where `T <: Vector{<:Real}` whose elements are stored in memory as a dense matrix. This approach has the advantage of making it completely explicit how Stheno will treat a matrix of data, and also simplifies quite a bit of the internal machinery, as all vectors of inputs can actually be assumed to be a subtype of `AbstractVector`.

Future plans include a `RowVecs` type which would instead treat each row of `X_data` as a vector-valued input. If you would like this feature, please raise an issue or a PR to let us know there's demand for it.


## Rectilinear Grids of Points in Euclidean Space

