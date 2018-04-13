import Base: +, *
export MeanFunction, CustomMean, ZeroMean, OneMean, ConstantMean, FiniteMean

"""
    MeanFunction
"""
abstract type MeanFunction end
length(::MeanFunction) = Inf
size(μ::MeanFunction) = (size(μ, 1),)
size(::MeanFunction, N::Int) = N == 1 ? Inf : 1

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
==(::ZeroMean{<:Any}, ::ZeroMean{<:Any}) = true

"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
mean(μ::ConstantMean, X::AVM) = fill(μ.c, size(X, 1))
==(μ::ConstantMean{<:Any}, μ′::ConstantMean{<:Any}) = μ.c == μ′.c 

# # Define composite mean functions.
# struct CompositeMean{O, T<:Tuple{Any, N} where N} <: μFun
#     args::T
# end

# +(μ::μFun, x′::Real) = μ + ConstantMean(x′)
# +(x::Real, μ′::μFun) = ConstantMean(x) + μ′
# +(μ::T, μ′::T′) where {T<:μFun, T′<:μFun} = CompositeMean{+, Tuple{T, T′}}((μ, μ′))
# (c::CompositeMean{+})(x) = c.args[1](x) + c.args[2](x)

# *(μ::μFun, x′::Real) = μ * ConstantMean(x′)
# *(x::Real, μ′::μFun) = ConstantMean(x) * μ′
# *(μ::T, μ′::T′) where {T<:μFun, T′<:μFun} = CompositeMean{*, Tuple{T, T′}}((μ, μ′))
# (c::CompositeMean{*})(x) = c.args[1](x) * c.args[2](x)
