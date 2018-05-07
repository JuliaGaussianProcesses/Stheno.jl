using FillArrays, Distances

import Base: +, *, ==
export KernelType, Kernel, EQ, RQ, Linear, Poly, Noise, Wiener, WienerVelocity, Exponential,
    ConstantKernel, isstationary, ZeroKernel

# Some fallback definitions.
cov(k::Kernel, X::AMRV) = pairwise(k, X)
cov(k::Kernel, X::AMRV, X′::AMRV) = pairwise(k, X, X′)
xcov(k::CrossKernel, X::AMRV) = pairwise(k, X)
xcov(k::CrossKernel, X::AMRV, X′::AMRV) = pairwise(k, X, X′)

# Some fallback definitions.
size(::CrossKernel, N::Int) = (N ∈ (1, 2)) ? Inf : 1
size(k::CrossKernel) = (size(k, 1), size(k, 2))
isstationary(::Type{<:CrossKernel}) = false
isstationary(::T) where T = isstationary(T)

"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
(::ZeroKernel{T})(x, x′) where T = zero(T)
isstationary(::Type{<:ZeroKernel}) = true
binary_colwise(::ZeroKernel{T}, X::AMRV, ::AMRV) where T = Zeros{T}(size(X, 2))
pairwise(::ZeroKernel{T}, X::AMRV, X′::AMRV) where T = Zeros{T}(size(X, 2), size(X′, 2))
==(::ZeroKernel{<:Any}, ::ZeroKernel{<:Any}) = true

"""
    ConstantKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct ConstantKernel{T<:Real} <: Kernel
    c::T
end
(k::ConstantKernel)(x, x′) = k.c
isstationary(::Type{<:ConstantKernel}) = true
binary_colwise(k::ConstantKernel, X::AMRV, ::AMRV) = Fill(k.c, size(X, 2))
pairwise(k::ConstantKernel, X::AMRV, Y::AMRV) = Fill(k.c, size(X, 2), size(Y, 2))
==(k::ConstantKernel, k′::ConstantKernel) = k.c == k′.c

"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel end
isstationary(::Type{<:EQ}) = true
(::EQ)(x, x′) = exp(-0.5 * sqeuclidean(x, x′))
pairwise(::EQ, X::AMRV) = exp.(-0.5 .* pairwise(SqEuclidean(), X))
pairwise(::EQ, X::AMRV, X′::AMRV) = exp.(-0.5 .* pairwise(SqEuclidean(), X, X′))

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
==(a::Linear, b::Linear) = a.c == b.c
(k::Linear)(x, x′) = dot(x .- k.c, x′ .- k.c)
function pairwise(k::Linear, X::AMRV)
    Δ = X .- k.c
    return Δ' * Δ
end
pairwise(k::Linear, X::AMRV, X′::AMRV) = (X .- k.c)' * (X′ .- k.c)

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

"""
    Noise{T<:Real} <: Kernel

A white-noise kernel with a single scalar parameter.
"""
struct Noise{T<:Real} <: Kernel
    σ²::T
end
isstationary(::Type{<:Noise}) = true
==(a::Noise, b::Noise) = a.σ² == b.σ²
(k::Noise)(x, x′) = x === x′ || x == x′ ? k.σ² : zero(k.σ²)
pairwise(k::Noise, X::AMRV) = k.σ² .* (pairwise(SqEuclidean(), X) .== 0)
pairwise(k::Noise, X::AMRV, X′::AMRV) = k.σ² .* (pairwise(SqEuclidean(), X, X′) .== 0)

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
