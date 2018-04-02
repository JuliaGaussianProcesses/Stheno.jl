import Base: +, *, ==, show
export KernelType, Kernel, EQ, RQ, Linear, Poly, Noise, Wiener, WienerVelocity, Exponential,
    Constant, isstationary

"""
    Kernel

Supertype for all kernels.
"""
abstract type Kernel end

isfinite(::Kernel) = false
isstationary(::Type{<:Kernel}) = false
isstationary(k::Kernel) = isstationary(typeof(k))

"""
    Zero <: Kernel

A rank 1 kernel that always returns zero.
"""
struct Zero <: Kernel end
(::Zero)(x::T, x′::T) where T = zero(T)
isstationary(::Type{<:Zero}) = true
show(io::IO, ::Zero) = show(io, "Zero")

"""
    Constant{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct Constant{T<:Real} <: Kernel
    value::T
end
(k::Constant)(::Any, ::Any) = k.value
==(a::Constant, b::Constant) = a.value == b.value
isstationary(::Type{<:Constant}) = true
show(io::IO, k::Constant) = show(io, "Constant($(k.value))")

"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel end
@inline (::EQ)(x::Real, y::Real) = exp(-0.5 * abs2(x - y))
isstationary(::Type{<:EQ}) = true
show(io::IO, ::EQ) = show(io, "EQ")

"""
    RQ{T<:Real} <: Kernel

The standardised Rational Quadratic. `RQ(α)` creates an `RQ` `Kernel{Stationary}` whose
kurtosis is `α`.
"""
struct RQ{T<:Real} <: Kernel
    α::T
end
@inline (k::RQ)(x::Real, y::Real) = (1 + 0.5 * abs2(x - y) / k.α)^(-k.α)
==(a::RQ, b::RQ) = a.α == b.α
isstationary(::Type{<:RQ}) = true
show(io::IO, k::RQ) = show(io, "RQ($(k.α))")

"""
    Linear{T<:Real} <: Kernel

Standardised linear kernel. `Linear(c)` creates a `Linear` `Kernel{NonStationary}` whose
intercept is `c`.
"""
struct Linear{T<:Real} <: Kernel
    c::T
end
@inline (k::Linear)(x, y) = dot(x - k.c, y - k.c)
@inline (k::Linear)(x::Tuple, y::Tuple) = sum(map((x, y)->(x - k.c) * (y - k.c), x, y))
==(a::Linear, b::Linear) = a.c == b.c
show(io::IO, k::Linear) = show(io, "Linear")

"""
    Poly{Tσ<:Real} <: Kernel

Standardised Polynomial kernel. `Poly(p, σ)` creates a `Poly`.
"""
struct Poly{Tσ<:Real} <: Kernel
    p::Int
    σ::Tσ
end
@inline (k::Poly)(x::Real, x′::Real) = (x * x′ + k.σ)^k.p
show(io::IO, k::Poly) = show(io, "Poly($(k.p))")

"""
    Noise <: Kernel

A standardised stationary white-noise kernel.
"""
struct Noise <: Kernel end
@inline (::Noise)(x::Real, x′::Real) = x == x′ ? 1.0 : 0.0
isstationary(::Type{<:Noise}) = true
show(io::IO, ::Noise) = show(io, "Noise")

"""
    Wiener <: Kernel

The standardised stationary Wiener-process kernel.
"""
struct Wiener <: Kernel end
@inline (::Wiener)(x::Real, x′::Real) = min(x, x′)
show(io::IO, ::Wiener) = show(io, "Wiener")

"""
    WienerVelocity <: Kernel

The standardised WienerVelocity kernel.
"""
struct WienerVelocity <: Kernel end
@inline (::WienerVelocity)(x::Real, x′::Real) =
    min(x, x′)^3 / 3 + abs(x - x′) * min(x, x′)^2 / 2
show(io::IO, ::WienerVelocity) = show(io, "WienerVelocity")

"""
    Exponential <: Kernel

The standardised Exponential kernel.
"""
struct Exponential <: Kernel end
@inline (::Exponential)(x::Real, x′::Real) = exp(-abs(x - x′))
isstationary(::Type{<:Exponential}) = true
show(io::IO, ::Exponential) = show(io, "Exp")
