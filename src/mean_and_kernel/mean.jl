export CustomMean

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
