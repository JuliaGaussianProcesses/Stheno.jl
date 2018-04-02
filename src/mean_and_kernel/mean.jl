import Base: +, *
export CustomMean, ZeroMean, OneMean, ConstantMean, FiniteMean

abstract type μFun end

# Allow a user to define a custom mean type.
struct CustomMean{F} <: μFun
    f::F
end
(μ::CustomMean)(x) = μ.f(x)

# The unary function which returns 0 everywhere.
struct ZeroMean <: μFun end
(::ZeroMean)(::Any) = 0.0

# The unary function which returns 1 everywhere.
struct OneMean <: μFun end
(::OneMean)(::Any) = 1.0

# A constant function.
struct ConstantMean{T} <: μFun
    c::T
end
(μ::ConstantMean)(::Any) = μ.c

# Define composite mean functions.
struct CompositeMean{O, T<:Tuple{Any, N} where N} <: μFun
    args::T
end

+(μ::μFun, x′::Real) = μ + ConstantMean(x′)
+(x::Real, μ′::μFun) = ConstantMean(x) + μ′
+(μ::T, μ′::T′) where {T<:μFun, T′<:μFun} = CompositeMean{+, Tuple{T, T′}}((μ, μ′))
(c::CompositeMean{+})(x) = c.args[1](x) + c.args[2](x)

*(μ::μFun, x′::Real) = μ * ConstantMean(x′)
*(x::Real, μ′::μFun) = ConstantMean(x) * μ′
*(μ::T, μ′::T′) where {T<:μFun, T′<:μFun} = CompositeMean{*, Tuple{T, T′}}((μ, μ′))
(c::CompositeMean{*})(x) = c.args[1](x) * c.args[2](x)
