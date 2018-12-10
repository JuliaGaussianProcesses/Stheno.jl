import Base: ==, map, AbstractVector, map, +, *, length, size, zero, iszero
import Statistics: mean

using Base.Broadcast: DefaultArrayStyle
import Base.Broadcast: broadcasted

export MeanFunction, CustomMean, ZeroMean, ConstantMean, mean

abstract type MeanFunction end
abstract type BaseMeanFunction <: MeanFunction end

"""
    AbstractVector(μ::MeanFunction)

Convert `μ` into an `AbstractVector` if such a representation exists.
"""
function AbstractVector(μ::MeanFunction)
    @assert isfinite(length(μ))
    return μ.(eachindex(μ))
end

eachindex(μ::BaseMeanFunction) = throw(ErrorException("Cannot construct indices for $μ"))
length(::BaseMeanFunction) = Inf
size(μ::MeanFunction) = (length(μ),)

"""
    CustomMean <: BaseMeanFunction

A user-defined mean function. `f(x)` should return a scalar for whatever type of `x` this is
intended to work with.
"""
struct CustomMean{T} <: BaseMeanFunction
    f::T
end
(f::CustomMean)(x) = f.f(x)

"""
    ZeroMean <: BaseMeanFunction

Returns zero (of the appropriate type) everywhere.
"""
struct ZeroMean{T<:Real} <: BaseMeanFunction end
(::ZeroMean{T})(x) where T = zero(T)
function broadcasted(::DefaultArrayStyle{1}, ::ZeroMean{T}, x::AbstractVector) where T
    return Zeros{T}(length(x))
end
==(::ZeroMean, ::ZeroMean) = true

const ZM = ZeroMean{Float64}
zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

+(x::ZeroMean, x′::ZeroMean) = zero(x)

"""
    ConstantMean{T} <: BaseMeanFunction

Returns `c` (of the appropriate type) everywhere.
"""
struct ConstantMean{T<:Real} <: BaseMeanFunction
    c::T
end
(μ::ConstantMean)(x) = μ.c
function broadcasted(::DefaultArrayStyle{1}, μ::ConstantMean, D::AbstractVector)
    return Fill(μ.c, length(D))
end
==(μ::ConstantMean, μ′::ConstantMean) = μ.c == μ′.c

+(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
*(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)

"""
    EmpiricalMean <: BaseMeanFunction

A finite-dimensional mean function specified by a vector of values `μ`.
"""
struct EmpiricalMean{Tμ<:AbstractVector} <: BaseMeanFunction
    μ::Tμ
end
(μ::EmpiricalMean)(n) = μ.μ[n]
==(μ1::EmpiricalMean, μ2::EmpiricalMean) = μ1.μ == μ2.μ
length(μ::EmpiricalMean) = length(μ.μ)
eachindex(μ::EmpiricalMean) = eachindex(μ.μ)
broadcasted(::DefaultArrayStyle{1}, μ::EmpiricalMean, ::Colon) = μ.μ
function broadcasted(::DefaultArrayStyle{1}, μ::EmpiricalMean, x::AV)
    return X == eachindex(μ) ? μ.μ : μ[x]
end
AbstractVector(μ::EmpiricalMean) = μ.μ

function broadcasted(::DefaultArrayStyle{1}, f::MeanFunction, x::BlockData)
    return BlockVector([f.(x) for x in blocks(x)])
end

broadcasted(::DefaultArrayStyle{1}, f::MeanFunction, ::Colon) = map(f, eachindex(f))
