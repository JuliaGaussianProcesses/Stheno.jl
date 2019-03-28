import Base: map, zero

abstract type MeanFunction end

# Allow `map` to be manually fused.
map(μ::MeanFunction, x::AV) = materialize(_map(μ, x))
@adjoint function map(μ::MeanFunction, x::AV)
    return Zygote.forward((μ, x)->materialize(_map(μ, x)), μ, x)
end

"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
ZeroMean() = ZeroMean{Int}()
_map(::ZeroMean{T}, x::AV) where T = Zeros{T}(length(x))
zero(::MeanFunction) = ZeroMean()

"""
    OneMean{T} <: MeanFunction

Return `one(T)` everywhere.
"""
struct OneMean{T<:Real} <: MeanFunction end
OneMean() = OneMean{Int}()
_map(::OneMean{T}, x::AV) where T = Ones{T}(length(x))

"""
    ConstMean{T} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end
_map(m::ConstMean, x::AV) = Fill(m.c, length(x))

"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end
_map(f::CustomMean, x::AV) = bcd(f.f, x)
