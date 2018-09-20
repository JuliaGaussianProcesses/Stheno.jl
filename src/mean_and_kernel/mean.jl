import Base: ==, map, AbstractVector, map, +, *, length, size, zero, iszero
import Statistics: mean

export CustomMean, ZeroMean, ConstantMean, mean

abstract type MeanFunction end
abstract type BaseMeanFunction <: MeanFunction end

"""
    AbstractVector(μ::MeanFunction)

Convert `μ` into an `AbstractVector` if such a representation exists.
"""
function AbstractVector(μ::MeanFunction)
    @assert isfinite(length(μ))
    return map(μ, eachindex(μ))
end

eachindex(μ::BaseMeanFunction) = throw(ErrorException("Cannot construct indices for $μ"))
length(::BaseMeanFunction) = Inf
size(μ::MeanFunction) = (length(μ),)

_map_fallback(f::MeanFunction, X::AV) = [f(x) for x in X]
_map(f::MeanFunction, X::AV) = _map_fallback(f, X)
map(f::MeanFunction, X::AV) = _map(f, X)
map(f::MeanFunction, X::BlockData) = BlockVector([map(f, x) for x in blocks(X)])
map(f::MeanFunction, ::Colon) = map(f, eachindex(f))

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
@inline _map(z::ZeroMean{T}, D::AbstractVector) where T = Zeros{T}(length(D))
==(::ZeroMean, ::ZeroMean) = true

const ZM = ZeroMean{Float64}
zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

"""
    ConstantMean{T} <: BaseMeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: BaseMeanFunction
    c::T
end
@inline (μ::ConstantMean)(x) = μ.c
@inline _map(μ::ConstantMean, D::AbstractVector) = Fill(μ.c, length(D))
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c

+(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
*(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)

"""
    EmpiricalMean <: BaseMeanFunction

A finite-dimensional mean function specified by a vector of values `μ`.
"""
struct EmpiricalMean{T<:Real, Tμ<:AbstractVector{T}} <: BaseMeanFunction
    μ::Tμ
    EmpiricalMean(μ::Tμ) where {T<:Real, Tμ<:AbstractVector{T}} = new{T, Tμ}(μ)
end
@inline (μ::EmpiricalMean)(n) = μ.μ[n]
==(μ1::EmpiricalMean, μ2::EmpiricalMean) = μ1.μ == μ2.μ
@inline length(μ::EmpiricalMean) = length(μ.μ)
@inline eachindex(μ::EmpiricalMean) = eachindex(μ.μ)

@inline map(μ::EmpiricalMean, ::Colon) = μ.μ
function _map(μ::EmpiricalMean, X::AV)
    if X == eachindex(μ)
        return μ.μ
    else
        return μ[X]
    end
end
AbstractVector(μ::EmpiricalMean) = μ.μ

+(x::ZeroMean, x′::ZeroMean) = zero(x)
function +(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    if iszero(μ)
        return μ′
    elseif iszero(μ′)
        return μ
    else
        return CompositeMean(+, μ, μ′)
    end
end
function *(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    return iszero(μ) || iszero(μ′) ? zero(μ) : CompositeMean(*, μ, μ′)
end
