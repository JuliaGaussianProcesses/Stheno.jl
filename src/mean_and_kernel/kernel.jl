using IterTools
import Base: +, *, ==, size, eachindex
import Distances: pairwise
export CrossKernel, Kernel, cov, xcov, EQ, RQ, Linear, Poly, Noise, Wiener, WienerVelocity,
    Exponential, ConstantKernel, isstationary, ZeroKernel



############################# Define CrossKernels and Kernels ##############################

abstract type CrossKernel end
abstract type Kernel <: CrossKernel end

@inline isstationary(::Type{<:CrossKernel}) = false
@inline isstationary(x::CrossKernel) = isstationary(typeof(x))

# Some fallback definitions.
size(::CrossKernel, N::Int) = (N ∈ (1, 2)) ? Inf : 1
size(k::CrossKernel) = (size(k, 1), size(k, 2))

eachindex(k::Kernel, N::Int) = eachindex(k)



################################# `map` implementations #################################

map(k::CrossKernel, X::BlockData) = BlockVector([map(k, x) for x in blocks(X)])
function map(k::CrossKernel, X::BlockData, X′::BlockData)
    return BlockVector([map(k, x, x′) for (x, x′) in zip(blocks(X), blocks(X′))])
end



################################# Pairwise implementations #################################

# Fallback implementation for `pairwise` with `DataSet`s.
@inline pairwise(f::CrossKernel, X::ADS) = pairwise(f, X, X)
@inline pairwise(f::Kernel, X::ADS) = LazyPDMat(pairwise(f, X, X))
@inline pairwise(f::CrossKernel, X::ADS, X′::ADS) = pairwise_fallback(f, X, X′)

@inline function pairwise_fallback(f, X::ADS, X′::ADS)
    return [f(X[p], X′[q]) for p in eachindex(X), q in eachindex(X′)]
end

# COMMITING TYPE PIRACY!
@inline pairwise(f::PreMetric, x::AbstractVector) = pairwise(f, RowVector(x))
@inline pairwise(f::PreMetric, x::AV, x′::AV) = pairwise(f, RowVector(x), RowVector(x′))

# Syntactic sugar for pairwise.
@inline pairwise(f::CrossKernel, X::AV{<:Real}) = pairwise(f, DataSet(X))
@inline function pairwise(f::CrossKernel, X::AV{<:Real}, X′::AV{<:Real})
    return pairwise(f, DataSet(X), DataSet(X′))
end

# Specialisations of `pairwise` + sugar for `BlockData` sets. Returns a `BlockArray`.
function pairwise(f::CrossKernel, X::BlockData, X′::BlockData)
    return BlockMatrix([pairwise(f, x, x′) for x in blocks(X), x′ in blocks(X′)])
end
function pairwise(f::CrossKernel, X::AbstractVector{<:ADS}, X′::AbstractVector{<:ADS})
    return pairwise(f, BlockData(X), BlockData(X′))
end

# Edge cases for interactions between vectors of data and data.
pairwise(f::CrossKernel, X::AbstractVector{<:ADS}, X′::ADS) = pairwise(f, X, [X′])
pairwise(f::CrossKernel, X::ADS, X′::AbstractVector{<:ADS}) = pairwise(f, [X], X′)



################################ Define some basic kernels #################################

"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
(::ZeroKernel{T})(x, x′) where T = zero(T)
(::ZeroKernel{T})(x) where T = zero(T)
isstationary(::Type{<:ZeroKernel}) = true
@inline map(::ZeroKernel{T}, X::DataSet, X′::DataSet) where T = Zeros{T}(length(X))
@inline function pairwise(::ZeroKernel{T}, X::DataSet, X′::DataSet) where T
    return Zeros{T}(length(X), length(X′))
end
==(::ZeroKernel{<:Any}, ::ZeroKernel{<:Any}) = true

"""
    ConstantKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct ConstantKernel{T<:Real} <: Kernel
    c::T
end
(k::ConstantKernel)(x::T, x′::T) where T = k.c
(k::ConstantKernel)(x) = k.c
isstationary(::Type{<:ConstantKernel}) = true
map(k::ConstantKernel, X::DataSet, ::DataSet) = Fill(k.c, length(X))
pairwise(k::ConstantKernel, X::DataSet, X′::DataSet) = Fill(k.c, length(X), length(X′))
==(k::ConstantKernel, k′::ConstantKernel) = k.c == k′.c

"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel end
isstationary(::Type{<:EQ}) = true
(::EQ)(x, x′) = exp(-0.5 * sqeuclidean(x, x′))
(::EQ)(x::T) where T = one(Float64)
pairwise(::EQ, X::DataSet) = LazyPDMat(exp.(-0.5 .* pairwise(SqEuclidean(), X.X)))
pairwise(::EQ, X::DataSet, X′::DataSet) = exp.(-0.5 .* pairwise(SqEuclidean(), X.X, X′.X))

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
(k::Linear)(x) = sum(abs2, x .- k.c)

@inline pairwise(k::Linear, x::VectorData) = pairwise(k, DataSet(RowVector(x.X)))
@inline function pairwise(k::Linear, x::VectorData, x′::VectorData)
    return pairwise(k, DataSet(RowVector(x.X)), DataSet(RowVector(x′.X)))
end

function pairwise(k::Linear, D::MatrixData)
    Δ = D.X .- k.c
    return LazyPDMat(Δ' * Δ)
end
pairwise(k::Linear, X::MatrixData, X′::MatrixData) = (X.X .- k.c)' * (X′.X .- k.c)

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
(k::Noise)(x) = k.σ²
pairwise(k::Noise, X::DataSet) = LazyPDMat(Diagonal(Fill(k.σ², length(X))))
function pairwise(k::Noise, X::DataSet, X′::DataSet)
    return X === X′ ? pairwise(k, X) : Zeros(length(X), length(X′))
end

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

"""
    EmpiricalKernel <: Kernel

A finite-dimensional kernel defined in terms of a PSD matrix `Σ`.
"""
struct EmpiricalKernel{T<:LazyPDMat} <: Kernel
    Σ::T
end
@inline (k::EmpiricalKernel)(q::Int, q′::Int) = k.Σ[q, q′]
@inline (k::EmpiricalKernel)(q::Int) = k.Σ[q, q]
@inline size(k::EmpiricalKernel, N::Int) = size(k.Σ, N)

function AbstractMatrix(k::Kernel)
    @assert isfinite(size(k, 1))
    return LazyPDMat(pairwise(k, eachindex(k, 1)))
end
function AbstractMatrix(k::CrossKernel)
    @assert isfinite(size(k, 1))
    @assert isfinite(size(k, 2))
    return pairwise(k, eachindex(k, 1), eachindex(k, 2))
end
