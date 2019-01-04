using Zygote: derivative

"""
    DerivativeMean{T∇<:MeanFunction} <: MeanFunction

"""
struct DerivativeMean{T∇<:MeanFunction} <: MeanFunction
    m::T∇
end
(μ::DerivativeMean)(x::Real) = derivative(x->μ.m(x), x)
_map(μ::DerivativeMean, x::AV{<:Real}) = bcd(μ, x)


"""
    DerivativeKernel{Tk<:Kernel} <: Kernel

"""
struct DerivativeKernel{Tk<:Kernel} <: Kernel
    k::Tk
end

# Binary methods.
(k::DerivativeKernel)(x::Real, x′::Real) = derivative(x′->derivative(x->k.k(x, x′), x), x′)
_map(k::DerivativeKernel, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)
_pw(k::DerivativeKernel, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′')

# Unary methods.
(k::DerivativeKernel)(x::Real) = k(x, x)
_map(k::DerivativeKernel, x::AV{<:Real}) = bcd(k, x)
_pw(k::DerivativeKernel, x::AV{<:Real}) = _pw(k, x, x)


"""
    DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeLhsCross)(x::Real, x′::Real) = derivative(x->k.k(x, x′), x)
_map(k::DerivativeLhsCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)
_pw(k::DerivativeLhsCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)


"""
    DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeRhsCross)(x::Real, x′::Real) = derivative(x′->k.k(x, x′), x′)
_map(k::DerivativeRhsCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)
_pw(k::DerivativeRhsCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)


"""
    DerivativeCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeCross)(x::Real, x′::Real) = derivative(x′->derivative(x->k.k(x, x′), x), x′)
_map(k::DerivativeCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)
_pw(k::DerivativeCross, x::AV{<:Real}, x′::AV{<:Real}) = bcd(k, x, x′)
