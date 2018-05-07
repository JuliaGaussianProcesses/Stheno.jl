using IterTools

import Base: length, size
import Distances: pairwise
import Base: ==

abstract type MeanFunction end
abstract type CrossKernel end
abstract type Kernel <: CrossKernel end

# Fallback implementations for `unary_colwise`.
@inline unary_colwise(f, X::AMRV) = unary_colwise_fallback(f, X)

unary_colwise(f, x::AbstractVector) = unary_colwise(f, RowVector(x))
unary_colwise_fallback(f, x::RowVector) = map(f, x')
unary_colwise_fallback(f, X::AbstractMatrix) = [f(view(X, :, n)) for n in 1:size(X, 2)]

# Fallback implementations for `binary_colwise`.
@inline binary_colwise(f, X::AMRV) =
    isstationary(f) ?
        Fill(f(X[:, 1], X[:, 1]), size(X, 2)) :
        binary_colwise(f, X, X)
@inline binary_colwise(f, X::AMRV, X′::AMRV) = binary_colwise_fallback(f, X, X′)

binary_colwise_fallback(f, x::RowVector, x′::RowVector) = map(f, x', x′')
binary_colwise_fallback(f, X::AbstractMatrix, X′::AbstractMatrix) =
    [f(view(X, :, n), view(X′, :, n)) for n in 1:size(X, 2)]

# Fallback implementation for `pairwise`.
@inline pairwise(f, X::AMRV) = pairwise(f, X, X)
@inline pairwise(f, X::AMRV, X′::AMRV) = pairwise_fallback(f, X, X′)

pairwise_fallback(f, x::RowVector, x′::RowVector) = broadcast(f, x', x′)
pairwise_fallback(f, X::AbstractMatrix, X′::AbstractMatrix) =
    [f(view(X, :, p), view(X′, :, q)) for p in 1:size(X, 2), q in 1:size(X′, 2)]
