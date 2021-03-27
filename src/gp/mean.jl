import Base: zero

abstract type MeanFunction end



"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
ZeroMean() = ZeroMean{Float64}()
elementwise(::ZeroMean{T}, x::AV) where T = zeros(T, length(x))
zero(::MeanFunction) = ZeroMean()



"""
    OneMean{T} <: MeanFunction

Return `one(T)` everywhere.
"""
struct OneMean{T<:Real} <: MeanFunction end
OneMean() = OneMean{Float64}()
elementwise(::OneMean{T}, x::AV) where T = ones(T, length(x))



"""
    ConstMean{T} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end
elementwise(m::ConstMean, x::AV) = fill(m.c, length(x))



"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end
elementwise(f::CustomMean, x::AV) = map(f.f, x)
