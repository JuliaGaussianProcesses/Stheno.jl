import Base: ==, map, AbstractVector, map, +, *, length, size, zero, iszero
import Statistics: mean

import Base.Broadcast: broadcasted, materialize
export MeanFunction, CustomMean, ZeroMean, ConstantMean, mean

abstract type MeanFunction end

eachindex(μ::MeanFunction) = throw(ErrorException("Cannot construct indices for $μ"))
length(::MeanFunction) = Inf
size(μ::MeanFunction) = (length(μ),)

# Mapping now allows for fused operations.
map(μ::MeanFunction, x::Union{AV, Colon}) = materialize(_map(μ, x))
map(μ::MeanFunction, x::BlockData) = BlockVector([map(μ, x) for x in blocks(x)])
_map(μ::MeanFunction, x) = broadcasted(μ, x)

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
==(::ZeroMean, ::ZeroMean) = true
_map(::ZeroMean{T}, x::AV) where T = Zeros{T}(length(x))


"""
    ConstantMean{T} <: MeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: MeanFunction
    c::T
end
(μ::ConstantMean)(x) = μ.c
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c
_map(μ::ConstantMean, D::AV) = Fill(μ.c, length(D))

struct CustomMean{T} <: MeanFunction
    f::T
end
@inline (f::CustomMean)(x) = f.f(x)


"""
    EmpiricalMean <: MeanFunction

A finite-dimensional mean function specified by a vector of values `μ`.
"""
struct EmpiricalMean{Tμ<:AbstractVector} <: MeanFunction
    μ::Tμ
end
(μ::EmpiricalMean)(n) = μ.μ[n]
==(μ₁::EmpiricalMean, μ₂::EmpiricalMean) = μ₁.μ == μ₂.μ
_map(μ::EmpiricalMean, x::AV) = x == eachindex(μ) ? μ.μ : μ[x]
length(μ::EmpiricalMean) = length(μ.μ)
eachindex(μ::EmpiricalMean) = eachindex(μ.μ)
AbstractVector(μ::EmpiricalMean) = μ.μ


struct UnaryMean{Top, Tμ} <: MeanFunction
    op::Top
    μ::Tμ
end
@inline (μ::UnaryMean)(x) = μ.op(μ.μ(x))
==(μ::UnaryMean, μ′::UnaryMean) = μ.op == μ′.op && μ.μ == μ′.μ
_map(μ::UnaryMean, x::AV) = broadcasted(μ.op, _map(μ.μ, x))


struct BinaryMean{Top, Tμ₁, Tμ₂} <: MeanFunction
    op::Top
    μ₁::Tμ₁
    μ₂::Tμ₂
end
(μ::BinaryMean)(x) = μ.op(μ.μ₁(x), μ.μ₂(x))
(μ::BinaryMean{<:Any, <:Real, <:Any})(x) = μ.op(μ.μ₁, μ.μ₂(x))
(μ::BinaryMean{<:Any, <:Any, <:Real})(x) = μ.op(μ.μ₁(x), μ.μ₂)
==(μ::BinaryMean, μ′::BinaryMean) = μ.op == μ′.op && μ.μ₁ == μ′.μ₁ && μ.μ₂ == μ′.μ₂
_map(μ::BinaryMean, x::AV) = broadcasted(μ.op, _map(μ.μ₁, x), _map(μ.μ₂, x))



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
