import Base: map

abstract type MeanFunction end

# Allow `map` to be manually fused.
map(μ::MeanFunction, x::AV) = materialize(_map(μ, x))


"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end
ZeroMean() = ZeroMean{Int}()
_map(::ZeroMean{T}, x::AV) where T = Zeros{T}(length(x))


"""
    OneMean{T} <: MeanFunction

Return `one(T)` everywhere.
"""
struct OneMean{T<:Real} <: MeanFunction end
OneMean() = OneMean{Int}()
_map(::OneMean{T}, x::AV) where T = Fill(one(T), length(x))


"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end
_map(f::CustomMean, x::AV) = broadcasted(f.f, x)
