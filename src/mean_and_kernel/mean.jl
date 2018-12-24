import Base: ==, map, AbstractVector, +, *, length, size, zero, iszero
import Statistics: mean

import Base.Broadcast: broadcasted, materialize
export MeanFunction, CustomMean, ZeroMean, ConstantMean, mean

abstract type MeanFunction end


# Mapping now allows for fused operations.
map(μ::MeanFunction, x::Union{AV, Colon}) = materialize(_map(μ, x))
map(μ::MeanFunction, x::BlockData) = BlockVector([map(μ, x) for x in blocks(x)])


"""
    AbstractVector(μ::MeanFunction)

Convert `μ` into an `AbstractVector` if such a representation exists.
"""
function AbstractVector(μ::MeanFunction)
    @assert isfinite(length(μ))
    return map(μ, :)
end


"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
(::ZeroMean{T})(x) where T = zero(T)
_map(::ZeroMean{T}, x::AV) where T = Zeros{T}(length(x))


"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
(μ::ConstantMean)(x) = μ.c
_map(μ::ConstantMean, D::AV) = Fill(μ.c, length(D))


"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end
(f::CustomMean)(x) = f.f(x)
_map(f::CustomMean, x::AV) = broadcasted(f, x)


"""
    EmpiricalMean <: MeanFunction

A finite-dimensional mean function specified by a vector of values `μ`.
"""
struct EmpiricalMean{Tμ<:AbstractVector} <: MeanFunction
    μ::Tμ
end
(μ::EmpiricalMean)(n) = μ.μ[n]
map(μ::EmpiricalMean, ::Colon) = μ.μ
AbstractVector(μ::EmpiricalMean) = μ.μ



############################### Operations on mean functions ###############################

import Base: zero, +, *

const ZM = ZeroMean{Float64}
zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

# Addition of mean functions.
function +(μ::MeanFunction, μ′::MeanFunction)
    if iszero(μ)
        return μ′
    elseif iszero(μ′)
        return μ
    else
        return BinaryMean(+, μ, μ′)
    end
end
+(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
+(a::Real, μ::MeanFunction) = UnaryMean(m->a + m, μ)
+(μ::MeanFunction, a::Real) = UnaryMean(m->m + a, μ)

# Product of mean functions.
function *(μ::MeanFunction, μ′::MeanFunction)
    if iszero(μ)
        return μ
    elseif iszero(μ′)
        return μ′
    else
        return BinaryMean(*, μ, μ′)
    end
end
*(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)
# *(a::Real, μ::MeanFunction) = UnaryMean(m->a * m, μ)
# *(μ::MeanFunction, a::Real) = UnaryMean(m->m * a, μ)

@inline *(a::Real, μ::MeanFunction) = BinaryMean(*, a, μ)
@inline *(μ::MeanFunction, a::Real) = BinaryMean(*, μ, a)
