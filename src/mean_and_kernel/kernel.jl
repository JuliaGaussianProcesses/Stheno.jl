import Base: +, *, ==
export KernelType, Kernel, EQ, RQ, Linear, Poly, Noise, Wiener, WienerVelocity, Exponential,
    ConstantKernel, isstationary, ZeroKernel, xcov

# Fallback definitions.
isfinite(::CrossKernel) = false
isstationary(::Type{<:CrossKernel}) = false
isstationary(k::CrossKernel) = isstationary(typeof(k))
cov(k::Kernel, X::AM) = StridedPDMatrix(xcov(k, X, X))
xcov(k::CrossKernel, X::AM) = xcov(k, X, X)

"""
    ZeroKernel <: Kernel

A rank 1 kernel that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
isstationary(::Type{<:ZeroKernel}) = true
show(io::IO, ::ZeroKernel) = show(io, "ZeroKernel")
(::ZeroKernel{T})(x, x′) where T = zero(T)
xcov(::ZeroKernel{T}, X::AM, X′::AM) where T = zeros(T, size(X, 1), size(X′, 1))
==(::ZeroKernel{<:Any}, ::ZeroKernel{<:Any}) = true

"""
    ConstantKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct ConstantKernel{T<:Real} <: Kernel
    c::T
end
isstationary(::Type{<:ConstantKernel}) = true
show(io::IO, k::ConstantKernel) = show(io, "ConstantKernel($(k.c))")
(k::ConstantKernel)(x::T, x′::T) where T = k.c
xcov(k::ConstantKernel, X::AM, X′::AM) = fill(k.c, size(X, 1), size(X′, 1))
==(k::ConstantKernel, k′::ConstantKernel) = k.c == k′.c

"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel end
isstationary(::Type{<:EQ}) = true
show(io::IO, ::EQ) = show(io, "EQ")
(::EQ)(x::T, x′::T) where T = exp(-0.5 * sqeuclidean(x, x′))
cov(::EQ, X::AM) = StridedPDMatrix(exp.(-0.5 * pairwise(SqEuclidean(), X')))
xcov(::EQ, X::AM, X′::AM) = exp.(-0.5 * pairwise(SqEuclidean(), X', X′'))

# """
#     RQ{T<:Real} <: Kernel

# The standardised Rational Quadratic. `RQ(α)` creates an `RQ` `Kernel{Stationary}` whose
# kurtosis is `α`.
# """
# struct RQ{T<:Real} <: Kernel
#     α::T
# end
# @inline (k::RQ)(x::Real, y::Real) = (1 + 0.5 * abs2(x - y) / k.α)^(-k.α)
# ==(a::RQ, b::RQ) = a.α == b.α
# isstationary(::Type{<:RQ}) = true
# show(io::IO, k::RQ) = show(io, "RQ($(k.α))")

"""
    Linear{T<:Real} <: Kernel

Standardised linear kernel. `Linear(c)` creates a `Linear` `Kernel{NonStationary}` whose
intercept is `c`.
"""
struct Linear{T<:Union{Real, Vector{<:Real}}} <: Kernel
    c::T
end
(k::Linear)(x, x′) = dot(x - k.c, x′ - k.c)
function cov(k::Linear, X::AM)
    Δ = X .- k.c
    return StridedPDMat(Δ * Δ')
end
xcov(k::Linear, X::AM, X′::AM) = (X .- k.c) * (X′ .- k.c)'
==(a::Linear, b::Linear) = a.c == b.c
show(io::IO, k::Linear) = show(io, "Linear")

# """
#     Poly{Tσ<:Real} <: Kernel

# Standardised Polynomial kernel. `Poly(p, σ)` creates a `Poly`.
# """
# struct Poly{Tσ<:Real} <: Kernel
#     p::Int
#     σ::Tσ
# end
# @inline (k::Poly)(x::Real, x′::Real) = (x * x′ + k.σ)^k.p
# show(io::IO, k::Poly) = show(io, "Poly($(k.p))")

# """
#     Noise <: Kernel

# A standardised stationary white-noise kernel.
# """
# struct Noise <: Kernel end
# @inline (::Noise)(x::Real, x′::Real) = x == x′ ? 1.0 : 0.0
# isstationary(::Type{<:Noise}) = true
# show(io::IO, ::Noise) = show(io, "Noise")

# """
#     Wiener <: Kernel

# The standardised stationary Wiener-process kernel.
# """
# struct Wiener <: Kernel end
# @inline (::Wiener)(x::Real, x′::Real) = min(x, x′)
# cov(::Wiener, X::AM, X′::AM) =
# show(io::IO, ::Wiener) = show(io, "Wiener")

# """
#     WienerVelocity <: Kernel

# The standardised WienerVelocity kernel.
# """
# struct WienerVelocity <: Kernel end
# @inline (::WienerVelocity)(x::Real, x′::Real) =
#     min(x, x′)^3 / 3 + abs(x - x′) * min(x, x′)^2 / 2
# show(io::IO, ::WienerVelocity) = show(io, "WienerVelocity")

# """
#     Exponential <: Kernel

# The standardised Exponential kernel.
# """
# struct Exponential <: Kernel end
# @inline (::Exponential)(x::Real, x′::Real) = exp(-abs(x - x′))
# isstationary(::Type{<:Exponential}) = true
# show(io::IO, ::Exponential) = show(io, "Exp")
