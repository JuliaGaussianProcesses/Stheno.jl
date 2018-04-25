import Base: +, *
export MeanFunction, CustomMean, ZeroMean, ConstantMean, FiniteMean

"""
    MeanFunction
"""
abstract type MeanFunction end
length(::MeanFunction) = Inf
size(μ::MeanFunction) = (size(μ, 1),)
size(μ::MeanFunction, N::Int) = N == 1 ? length(μ) : 1

"""
    CustomMean <: MeanFunction

A user-defined mean function. `f` should be defined such that when applied to a `N x D`
(`Abstract`)`Matrix`, an `N`-`AbstractVector` is returned.
"""
struct CustomMean{T} <: MeanFunction
    f::T
end
mean(μ::CustomMean, X::AVM) = μ.f(X)

"""
    ZeroMean <: MeanFunction

Returns zero (of the appropriate type) everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
mean(::ZeroMean{T}, X::AVM) where T = zeros(T, size(X, 1))
==(::ZeroMean, ::ZeroMean) = true

"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
function mean(μ::ConstantMean{T}, X::AVM) where T
    v = fill(one(T), size(X, 1))
    return μ.c .* v
end
# mean(μ::ConstantMean, X::AVM) = fill(μ.c, size(X, 1))
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c 
