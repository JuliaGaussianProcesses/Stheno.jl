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
length(k::Kernel) = size(k, 1)

eachindex(k::Kernel, N::Int) = eachindex(k)



###################### `map` and `pairwise` fallback implementations #######################

@inline _map_fallback(k::CrossKernel, X::AV) = [k(x, x) for x in X]
@inline _map_fallback(k::Kernel, X::AV) = [k(x) for x in X]
@inline _map(k::CrossKernel, X::AV) = _map_fallback(k, X)
@inline map(k::CrossKernel, X::AV) = _map(k, X)
map(k::CrossKernel, X::BlockData) = BlockVector([map(k, x) for x in blocks(X)])

@inline _map_fallback(k::CrossKernel, X::AV, X′::AV) = [k(x, x′) for (x, x′) in zip(X, X′)]
@inline _map(k::CrossKernel, X::AV, X′::AV) = _map_fallback(k, X, X′)
@inline map(k::CrossKernel, X::AV, X′::AV) = _map(k, X, X′)
function map(k::CrossKernel, X::BlockData, X′::BlockData)
    return BlockVector([map(k, x, x′) for (x, x′) in zip(blocks(X), blocks(X′))])
end
map(k::CrossKernel, X::BlockData, X′::AV) = map(k, X, BlockData([X′]))
map(k::CrossKernel, X::AV, X′::BlockData) = map(k, BlockData([X]), X′)

function _pairwise_fallback(k::CrossKernel, X::AV, X′::AV)
    return [k(X[p], X′[q]) for p in eachindex(X), q in eachindex(X′)]
end
_pairwise(k::CrossKernel, X::AV, X′::AV) = _pairwise_fallback(k, X, X′)
pairwise(k::CrossKernel, X::AV, X′::AV) = _pairwise(k, X, X′)
function pairwise(k::CrossKernel, X::BlockData, X′::BlockData)
    return BlockMatrix([pairwise(k, x, x′) for x in blocks(X), x′ in blocks(X′)])
end
pairwise(k::CrossKernel, X::BlockData, X′::AV) = pairwise(k, X, BlockData([X′]))
pairwise(k::CrossKernel, X::AV, X′::BlockData) = pairwise(k, BlockData([X]), X′)
pairwise(k::CrossKernel, X::AV) = pairwise(k, X, X)

_pairwise(k::Kernel, X::AV) = _pairwise(k, X, X)
pairwise(k::Kernel, X::AV) = LazyPDMat(_pairwise(k, X))
function pairwise(k::Kernel, X::BlockData)
    return LazyPDMat(BlockMatrix([pairwise(k, x, x′) for x in blocks(X), x′ in blocks(X)]))
end

# Sugar for `eachindex` things.
for op in [:map, :pairwise]
    @eval begin
        $op(k::CrossKernel, ::Colon) = $op(k, eachindex(k))
        $op(k::CrossKernel, ::Colon, ::Colon) = $op(k, eachindex(k, 1), eachindex(k, 2))
        $op(k::CrossKernel, ::Colon, X′::AV) = $op(k, eachindex(k, 1), X′)
        $op(k::CrossKernel, X::AV, ::Colon) = $op(k, X, eachindex(k, 2))
    end
end


################################ Define some basic kernels #################################

"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
(::ZeroKernel{T})(x, x′) where T = zero(T)
(::ZeroKernel{T})(x) where T = zero(T)
isstationary(::Type{<:ZeroKernel}) = true
@inline _map(::ZeroKernel{T}, X::AV, X′::AV) where T = Zeros{T}(length(X))
@inline function _pairwise(::ZeroKernel{T}, X::AV, X′::AV) where T
    return Zeros{T}(length(X), length(X′))
end
==(::ZeroKernel{<:Any}, ::ZeroKernel{<:Any}) = true

# ZeroKernel-specific optimisations.
+(k::CrossKernel, k′::ZeroKernel) = k
+(k::ZeroKernel, k′::CrossKernel) = k′
+(k::ZeroKernel, k′::ZeroKernel) = k
*(k::CrossKernel, k′::ZeroKernel) = k′
*(k::ZeroKernel, k′::CrossKernel) = k
*(k::ZeroKernel, k′::ZeroKernel) = k

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
_map(k::ConstantKernel, X::AV, ::AV) = Fill(k.c, length(X))
_pairwise(k::ConstantKernel, X::AV, X′::AV) = Fill(k.c, length(X), length(X′))
==(k::ConstantKernel, k′::ConstantKernel) = k.c == k′.c

# ConstantKernel-specific optimisations.
+(k::ConstantKernel, k′::ConstantKernel) = ConstantKernel(k.c + k′.c)
*(k::ConstantKernel, k′::ConstantKernel) = ConstantKernel(k.c * k′.c)

"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel end
isstationary(::Type{<:EQ}) = true
(::EQ)(x, x′) = exp(-0.5 * sqeuclidean(x, x′))
(::EQ)(x::T) where T = one(Float64)
_pairwise(::EQ, X::MatData) = exp.(-0.5 .* pairwise(SqEuclidean(), X.X))
_pairwise(::EQ, X::MatData, X′::MatData) = exp.(-0.5 .* pairwise(SqEuclidean(), X.X, X′.X))

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

@inline _pairwise(k::Linear, x::AV) = _pairwise(k, MatData(RowVector(x)))
@inline function _pairwise(k::Linear, x::AV, x′::AV)
    return _pairwise(k, MatData(RowVector(x)), MatData(RowVector(x′)))
end

function _pairwise(k::Linear, D::MatData)
    Δ = D.X .- k.c
    return Δ' * Δ
end
_pairwise(k::Linear, X::MatData, X′::MatData) = (X.X .- k.c)' * (X′.X .- k.c)

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
_pairwise(k::Noise, X::AV) = Diagonal(Fill(k.σ², length(X)))
function _pairwise(k::Noise, X::AV, X′::AV)
    if X === X′
        return _pairwise(k, X)
    else
        return [view(X, p) == view(X′, q) ? k.σ² : 0
            for p in eachindex(X), q in eachindex(X′)]
    end
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
