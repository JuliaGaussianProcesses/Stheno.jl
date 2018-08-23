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
function (k::DerivativeKernel)(x::Real, x′::Real)
    return derivative(x′->derivative(x->k.k(x, x′), x), x′)
end
(k::DerivativeKernel)(x::Real) = k(x, x)

"""
    DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeLhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
function (k::DerivativeLhsCross)(x::Real, x′::Real)
    return derivative(x->k.k(x, x′), x)
end

"""
    DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeRhsCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
function (k::DerivativeRhsCross)(x::Real, x′::Real)
    return derivative(x′->k.k(x, x′), x′)
end

"""
    DerivativeCross{Tk<:CrossKernel} <: CrossKernel

"""
struct DerivativeCross{Tk<:CrossKernel} <: CrossKernel
    k::Tk
end
function (k::DerivativeCross)(x::Real, x′::Real)
    return derivative(x′->derivative(x->k.k(x, x′), x), x′)
end
(k::DerivativeCross)(x::Real) = k(x, x)
