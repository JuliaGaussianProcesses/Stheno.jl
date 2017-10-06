import Base: +, *, ==
export KernelType, Kernel, EQ, RQ, Linear, Circularised, lb, ub

"""
Determines whether a kernel is stationary or not and enables dispatch on this.
"""
abstract type KernelType end

""" Indicates that a `Kernel` is non-stationary. """
abstract type NonStationary <: KernelType end

""" Indicates that a `Kernel` is stationary. """
abstract type Stationary <: KernelType end

"""
    Kernel{T<:KernelType}

Supertype for all kernels; `T` indicates whether is `Stationary` or `NonStationary`.
"""
abstract type Kernel{T<:KernelType} end

"""
    Constant{T<:Real} <: Kernel{Stationary}

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct Constant{T<:Real} <: Kernel{Stationary}
    value::T
end
(k::Constant)(::Any, ::Any) = k.value
==(a::Constant, b::Constant) = a.value == b.value

"""
    EQ <: Kernel{Stationary}

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel{Stationary} end
(::EQ)(x::Real, y::Real) = exp(-0.5 * abs2(x - y))
==(::EQ, ::EQ) = true

"""
    RQ{T<:Real} <: Kernel{Stationary}

The standardised Rational Quadratic. `RQ(α)` creates an `RQ` `Kernel{Stationary}` whose
kurtosis is `α`.
"""
struct RQ{T<:Real} <: Kernel{Stationary}
    α::T
end
(k::RQ)(x::Real, y::Real) = (1 + 0.5 * abs2(x - y) / k.α)^(-k.α)
==(a::RQ, b::RQ) = a.α == b.α

"""
    Linear{T<:Real} <: Kernel{NonStationary}

Standardised linear kernel. `Linear(c)` creates a `Linear` `Kernel{NonStationary}` whose
intercept is `c`.
"""
struct Linear{T<:Real} <: Kernel{NonStationary}
    c::T
end
(k::Linear)(x::Real, y::Real) = (x - k.c) * (y - k.c)
==(a::Linear, b::Linear) = a.c == b.c
