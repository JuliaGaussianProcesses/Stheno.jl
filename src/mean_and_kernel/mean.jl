import Base: mean, ==

export CustomMean, ZeroMean, ConstantMean, mean

abstract type BaseMeanFunction <: MeanFunction end

eachindex(μ::BaseMeanFunction) = throw(ErrorException("Cannot construct indices for $μ"))
length(::BaseMeanFunction) = Inf
size(μ::BaseMeanFunction) = (size(μ, 1),)
size(μ::BaseMeanFunction, N::Int) = N == 1 ? length(μ) : 1

"""
    CustomMean <: BaseMeanFunction

A user-defined mean function. `f(x)` should return a scalar for whatever type of `x` this is
intended to work with.
"""
struct CustomMean{T} <: BaseMeanFunction
    f::T
end
@inline (f::CustomMean)(x) = f.f(x)

"""
    ZeroMean <: BaseMeanFunction

Returns zero (of the appropriate type) everywhere.
"""
struct ZeroMean{T<:Real} <: BaseMeanFunction end
@inline (::ZeroMean{T})(x) where T = zero(T)
@inline unary_obswise(z::ZeroMean{T}, X::AVM) where T = Zeros{T}(nobs(X))
==(::ZeroMean, ::ZeroMean) = true

"""
    ConstantMean{T} <: BaseMeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: BaseMeanFunction
    c::T
end
@inline (μ::ConstantMean)(x) = μ.c
@inline unary_obswise(μ::ConstantMean, X::AVM) = Fill(μ.c, nobs(X))
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c 
