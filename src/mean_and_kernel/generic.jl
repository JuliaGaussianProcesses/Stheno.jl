using FillArrays, IterTools

import Base: length, size
import Distances: pairwise
import Base: ==

abstract type MeanFunctionOrKernel <: Function end
abstract type MeanFunction <: MeanFunctionOrKernel end
abstract type Kernel <: MeanFunctionOrKernel end

# Fallback implementations for `unary_colwise`.
@inline unary_colwise(f, X::AMRV) = unary_colwise_fallback(f, X)

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

isstationary(::Type{<:MeanFunctionOrKernel}) = false
isstationary(k::MeanFunctionOrKernel) = isstationary(typeof(k))

"""
    Zero{N, T<:Real} <: MeanFunctionOrKernel

Represents mean function and kernels with a `zero(T)` value. `N` is the arity.
"""
struct Zero{N, T<:Real} <: MeanFunctionOrKernel end
@inline (::Zero{1, T})(x) where T = zero(T)
@inline (::Zero{2, T})(x, x′) where T = zero(T)
@inline unary_colwise(z::Zero{1, T}, X::AMRV) where T = binary_colwise(Zero{2, T}(), X, X)
binary_colwise(::Zero{2, T}, X::AMRV, ::AMRV) where T = Zeros{T}(size(X, 2))
pairwise(::Zero{2, T}, X::AMRV, X′::AMRV) where T = Zeros{T}(size(X, 2), size(X′, 2))
isstationary(::Zero) = true
==(::Zero{N, T}, ::Zero{N, V}) where {N, T, V} = true

"""
    Const{N, T<:Real} <: MeanFunctionOrKernel

Represents mean functions and kernels with a constant value.
"""
struct Const{N, T<:Real} <: MeanFunctionOrKernel
    c::T
    Const(N::Int, c::T) where T<:Real = new{N, T}(c)
end
@inline (c::Const{1})(x) = c.c
@inline (c::Const{2})(x, x′) = c.c
@inline unary_colwise(c::Const{1}, X::AMRV) = binary_colwise(Const(2, c.c), X, X)
binary_colwise(c::Const{2}, X::AMRV, ::AMRV) = Fill(c.c, size(X, 2))
pairwise(c::Const{2}, X::AMRV, Y::AMRV) = Fill(c.c, size(X, 2), size(Y, 2))
isstationary(::Const) = true
==(c::Const{N}, c′::Const{N}) where N = c.c == c′.c

# Hand-code size information as we can't fallback.
for T in [:Zero, :Const]
@eval begin

length(x::$T{1}) = size(x, 1)
size(::$T{1}, N::Int) = N == 1 ? Inf : 1
size(::$T{2}, N::Int) = (N ∈ (1, 2)) ? Inf : 1

size(z::$T{1}) = (size(z, 1),)
size(z::$T{2}) = (size(z, 1), size(z, 2))

end

end
