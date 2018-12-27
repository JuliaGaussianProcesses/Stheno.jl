import Base: ==, map, AbstractVector, +, *, zero, iszero
import Statistics: mean

import Base.Broadcast: broadcasted, materialize
export MeanFunction, CustomMean, ZeroMean, OneMean, mean

abstract type MeanFunction end

# Allow `map` to be fused.
map(μ::MeanFunction, x::Union{AV, Colon}) = materialize(_map(μ, x))


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
ZeroMean() = ZeroMean{Int}()
(::ZeroMean{T})(x) where {T} = zero(T)
_map(::ZeroMean{T}, x::AV) where T = Zeros{T}(length(x))

@adjoint (m::ZeroMean{T})(x) where {T} = m(x), Δ->(zero(x),)
@adjoint function _map(m::ZeroMean{T}, x::AV) where {T}
    return _map(m, x), Δ->(nothing, Zeros{T}(length(x)))
end


"""
    OneMean{T} <: MeanFunction

Return `one(T)` everywhere.
"""
struct OneMean{T<:Real} <: MeanFunction end
OneMean() = OneMean{Int}()
(::OneMean{T})(x) where {T} = one(T)
_map(::OneMean{T}, x::AV) where T = Fill(one(T), length(x))

@adjoint (m::OneMean{T})(x) where {T} = m(x), Δ->(zero(x),)
@adjoint function _map(μ::OneMean{T}, x::AV) where {T}
    return _map(μ, x), Δ->(nothing, Zeros{eltype(T)}(length(x)))
end


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
