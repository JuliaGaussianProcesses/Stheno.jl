import Base: zero

abstract type MeanFunction <: AbstractModel end



"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
get_iparam(::ZeroMean) = Union{}[]
child(::ZeroMean) = ()
ZeroMean() = ZeroMean{Float64}()
ew(::ZeroMean{T}, x::AV) where T = zeros(T, length(x))
zero(::MeanFunction) = ZeroMean()



"""
    OneMean{T} <: MeanFunction

Return `one(T)` everywhere.
"""
struct OneMean{T<:Real} <: MeanFunction end
get_iparam(::OneMean) = Union{}[]
child(::OneMean) = ()
OneMean() = OneMean{Float64}()
ew(::OneMean{T}, x::AV) where T = ones(T, length(x))



"""
    ConstMean{T} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T, cT<:AV{T}} <: MeanFunction
    c::cT
end
get_iparam(c::ConstMean) = c.c
child(::ConstMean) = ()
ew(m::ConstMean, x::AV) = fill(m.c, length(x))



"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end
ew(f::CustomMean, x::AV) = map(f.f, x)
