import Base: mean, ==

export CustomMean, ZeroMean, ConstantMean, mean

length(::MeanFunction) = Inf
size(μ::MeanFunction) = (size(μ, 1),)
size(μ::MeanFunction, N::Int) = N == 1 ? length(μ) : 1

"""
    CustomMean <: MeanFunction

A user-defined mean function. `f(x)` should return a scalar for whatever type of `x` this is
intended to work with.
"""
struct CustomMean{T} <: MeanFunction
    f::T
end
@inline (f::CustomMean)(x) = f.f(x)

"""
    ZeroMean <: MeanFunction

Returns zero (of the appropriate type) everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
@inline (::ZeroMean{T})(x) where T = zero(T)
@inline unary_obswise(z::ZeroMean{T}, X::AVM) where T = Zeros{T}(nobs(X))
==(::ZeroMean, ::ZeroMean) = true

"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
@inline (μ::ConstantMean)(x) = μ.c
@inline unary_obswise(μ::ConstantMean, X::AVM) = Fill(μ.c, nobs(X))
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c 
