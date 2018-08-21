using ForwardDiff: derivative

"""
    DerivativeMean{T∇<:MeanFunction} <: MeanFunction

"""
struct DerivativeMean{T∇<:MeanFunction} <: MeanFunction
    m::T∇
end
(μ::DerivativeMean)(x::Real) = derivative(x->μ.m(x), x)

"""
    DerivativeKernel{Tk<:Kernel} <: Kernel

"""
struct DerivativeKernel{Tk<:Kernel} <: Kernel
    k::Tk
end
(k::DerivativeKernel)(x::Real, x′::Real) = derivative(x′->derivative(x->k.k(x, x′), x), x′)
(k::DerivativeKernel)(x::Real) = k(x, x)

"""
    DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeLhsCross)(x::Real, x′::Real) = derivative(x->k.k(x, x′), x)

"""
    DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeRhsCross)(x::Real, x′::Real) = derivative(x′->k.k(x, x′), x′)

"""
    DerivativeCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
(k::DerivativeCross)(x::Real, x′::Real) = derivative(x′->derivative(x->k.k(x, x′), x), x′)
(k::DerivativeCross)(x::Real) = k(x, x)
