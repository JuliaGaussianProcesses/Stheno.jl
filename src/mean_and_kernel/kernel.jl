using LinearAlgebra, GPUArrays
using Base.Broadcast: DefaultArrayStyle
using GPUArrays: GPUVector

import LinearAlgebra: AbstractMatrix, AdjOrTransAbsVec, AdjointAbsVec
import Base: +, *, ==, size, eachindex, print
import Distances: pairwise, colwise, sqeuclidean, SqEuclidean
import Base.Broadcast: broadcasted, materialize

const bcd = broadcasted

export CrossKernel, Kernel, cov, xcov, EQ, PerEQ, RQ, Linear, Poly, Noise, Wiener,
    WienerVelocity, Exponential, ConstantKernel, isstationary, ZeroKernel, pairwise



############################# Define CrossKernels and Kernels ##############################

abstract type CrossKernel end
abstract type Kernel <: CrossKernel end

"""
    map(k, x::AV)

map `k` over `x`, with the convention that `k(x) := k(x, x)`.
"""
map(k::CrossKernel, x::AV) = materialize(_map(k, x))
map(k::CrossKernel, x::GPUVector) = materialize(_map(k, x))

"""
    map(k::CrossKernel, x::AV, x′::AV)

map `k` over the elements of `x` and `x′`.
"""
map(k::CrossKernel, x::AV, x′::AV) = materialize(_map(k, x, x′))
map(k::CrossKernel, x::GPUVector, x′::GPUVector) = materialize(_map(k, x, x′))

"""
    pairwise(f, x::AV)

Compute the `length(x) × length(x′)` matrix whose `(p, q)`th element is `k(x[p], x[q])`.
`_pairwise` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV) = materialize(_pairwise(k, x))

"""
    pairwise(f, x::AV, x′::AV)

Compute the `length(x) × length(x′)` matrix whose `(p, q)`th element is `k(x[p], x′[q])`.
`_pairwise` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV, x′::AV) = materialize(_pairwise(k, x, x′))



################################ Util. for Toeplitz matrices ###############################

function toep_pw(k::CrossKernel, x::StepRangeLen, x′::StepRangeLen)
    if x.step == x′.step
        return Toeplitz(
            map(k, x, Fill(x′[1], length(x))),
            map(k, Fill(x[1], length(x′)), x′),
        )
    else
        return invoke(_pairwise, Tuple{typeof(k), AV, AV}, k, x, x′)
    end
end

toep_pw(k::Kernel, x::StepRangeLen) = SymmetricToeplitz(map(k, x, Fill(x[1], length(x))))

function toep_map(k::Kernel, x::StepRangeLen, x′::StepRangeLen)
    if x.step == x′.step
        return Fill(k(x[1], x′[1]), broadcast_shape(size(x), size(x′)))
    else
        return invoke(_map, Tuple{typeof(k), AV, AV}, k, x, x′)
    end
end



################################ Define some basic kernels #################################


"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end

# Binary methods.
(::ZeroKernel{T})(x, x′) where T = zero(T)
_map(::ZeroKernel{T}, x::AV, x′::AV) where T = Zeros{T}(broadcast_shape(size(x), size(x′)))
_pairwise(k::ZeroKernel{T}, x::AV, x′::AV) where T = Zeros{T}(length(x), length(x′))

# Unary methods.
(::ZeroKernel{T})(x) where T = zero(T)
_map(::ZeroKernel{T}, x::AV) where T = Zeros{T}(length(x))
_pairwise(k::ZeroKernel{T}, x::AV) where T = Zeros{T}(length(x), length(x))


"""
    ConstantKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct ConstantKernel{T<:Real} <: Kernel
    c::T
end

# Binary methods.
(k::ConstantKernel)(x, x′) = k(x)
_map(k::ConstantKernel, x::AV, x′::AV) = Fill(k.c, broadcast_shape(size(x), size(x′)))
_pairwise(k::ConstantKernel, x::AV, x′::AV) = Fill(k.c, length(x), length(x′))

# Unary methods.
(k::ConstantKernel)(x) = k.c
_map(k::ConstantKernel, x::AV) = Fill(k.c, length(x))
_pairwise(k::ConstantKernel, x::AV) = Fill(k.c, length(x), length(x))


"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel (no free parameters).
"""
struct EQ <: Kernel end

# Binary methods.
(::EQ)(x, x′) = exp(-sqeuclidean(x, x′) / 2)
function _map(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd((x, x′)->-sqeuclidean(x, x′) / 2, x, x′))
end
function _pairwise(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd((x, x′)->-sqeuclidean(x, x′) / 2, x, x′'))
end
function _map(::EQ, X::ColsAreObs, X′::ColsAreObs)
    return bcd(x->exp(-x / 2), colwise(SqEuclidean(), X.X, X′.X))
end
function _pairwise(::EQ, X::ColsAreObs, X′::ColsAreObs)
    return bcd(x->exp(-x / 2), pairwise(SqEuclidean(), X.X, X′.X))
end
_map(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pairwise(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
(::EQ)(x::Real) = one(x)
(::EQ)(x::AV{<:Real}) = one(eltype(x))
_map(::EQ, x::AV) = Ones{eltype(x)}(length(x))
_map(::EQ, X::ColsAreObs) = Ones{eltype(X.X)}(length(X))
_pairwise(::EQ, x::AV{<:Real}) = _pairwise(EQ(), x, x)
_pairwise(::EQ, X::ColsAreObs) = bcd(x->exp(-x / 2), pairwise(SqEuclidean(), X.X))
_pairwise(::EQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

# Optimised adjoints. These really do count in terms of performance.
@adjoint function(::EQ)(x::Real, x′::Real)
    s = EQ()(x, x′)
    return s, function(Δ)
        x̄′ = Δ * (x - x′) * s
        return -x̄′, x̄′
    end
end
@adjoint function _map(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = materialize(_map(EQ(), x, x′))
    return s, function(Δ)
        x̄′ = (x .- x′) .* Δ .* s
        return nothing, -x̄′, x̄′
    end
end
@adjoint function _pairwise(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = materialize(_pairwise(EQ(), x, x′))
    return s, function(Δ)
        x̄′ = Δ .* (x .- x′') .* s
        return nothing, -reshape(sum(x̄′; dims=2), :), reshape(sum(x̄′; dims=1), :)
    end
end
@adjoint function _pairwise(::EQ, x::AV{<:Real})
    s = materialize(_pairwise(EQ(), x))
    return s, function(Δ)
        x̄_tmp = Δ .* (x .- x') .* s
        return nothing, reshape(sum(x̄_tmp; dims=1), :) - reshape(sum(x̄_tmp; dims=2), :)
    end
end
@adjoint (::EQ)(x::Real) = (EQ()(x), _->(zero(x),))
@adjoint _map(::EQ, x::AV) = (_map(EQ(), x), _->(nothing, Zeros{eltype(x)}(length(x)),))


"""
    PerEQ

The usual periodic kernel derived by mapping the input domain onto the unit circle.
"""
struct PerEQ <: Kernel end

# Binary methods.
(::PerEQ)(x::Real, x′::Real) = exp(-2 * sin(π * abs(x - x′))^2)
function _map(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd(x->-2x^2, bcd(sin, bcd((x, x′)->π * abs(x - x′), x, x′))))
end
function _pairwise(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd(x->-2x^2, bcd(sin, bcd((x, x′)->π * abs(x - x′), x, x′'))))
end
_map(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pairwise(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
(::PerEQ)(x::Real) = one(typeof(x))
_map(::PerEQ, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pairwise(k::PerEQ, x::AV{<:Real}) = _pairwise(k, x, x)
_pairwise(::PerEQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

@adjoint (k::PerEQ)(x::Real) = (k(x), _->(zero(typeof(x)),))
@adjoint function _map(k::PerEQ, x::AV{<:Real})
    return _map(k, x), _->(nothing, Zeros{eltype(x)}(length(x),))
end


"""
    Exponential <: Kernel

The standardised Exponential kernel.
"""
struct Exponential <: Kernel end

# Binary methods
(::Exponential)(x::Real, x′::Real) = exp(-abs(x - x′))
function _map(k::Exponential, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′))
end
function _pairwise(k::Exponential, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′'))
end

# Unary methods
(::Exponential)(x::Real) = one(x)
_map(k::Exponential, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pairwise(k::Exponential, x::AV{<:Real}) = _pairwise(k, x, x)


"""
    Linear{T<:Real} <: Kernel

Standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

# Binary methods
(k::Linear)(x, x′) = dot(x .- k.c, x′ .- k.c)
_pairwise(k::Linear, x::AV, x′::AV) = _pairwise(k, ColsAreObs(x'), ColsAreObs(x′'))
_pairwise(k::Linear, X::ColsAreObs, X′::ColsAreObs) = (X.X .- k.c)' * (X′.X .- k.c)

# Unary methods
(k::Linear)(x) = sum(abs2, x .- k.c)
_pairwise(k::Linear, x::AV) = _pairwise(k, ColsAreObs(x'))
function _pairwise(k::Linear, D::ColsAreObs)
    Δ = D.X .- k.c
    return Δ' * Δ
end


"""
    Noise{T<:Real} <: Kernel

A white-noise kernel with a single scalar parameter.
"""
struct Noise{T<:Real} <: Kernel
    σ²::T
end
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
#     RQ{T<:Real} <: Kernel

# The standardised Rational Quadratic. `RQ(α)` creates an `RQ` `Kernel{Stationary}` whose
# kurtosis is `α`.
# """
# struct RQ{T<:Real} <: Kernel
#     α::T
# end
# @inline (k::RQ)(x::Real, y::Real) = (1 + 0.5 * abs2(x - y) / k.α)^(-k.α)


# """
#     Poly{Tσ<:Real} <: Kernel

# Standardised Polynomial kernel. `Poly(p, σ)` creates a `Poly`.
# """
# struct Poly{Tσ<:Real} <: Kernel
#     p::Int
#     σ::Tσ
# end
# @inline (k::Poly)(x::Real, x′::Real) = (x * x′ + k.σ)^k.p


# """
#     Wiener <: Kernel

# The standardised stationary Wiener-process kernel.
# """
# struct Wiener <: Kernel end
# @inline (::Wiener)(x::Real, x′::Real) = min(x, x′)
# cov(::Wiener, X::AM, X′::AM) =


# """
#     WienerVelocity <: Kernel

# The standardised WienerVelocity kernel.
# """
# struct WienerVelocity <: Kernel end
# @inline (::WienerVelocity)(x::Real, x′::Real) =
#     min(x, x′)^3 / 3 + abs(x - x′) * min(x, x′)^2 / 2


"""
    EmpiricalKernel <: Kernel

A finite-dimensional kernel defined in terms of a PSD matrix `Σ`.
"""
struct EmpiricalKernel{T<:LazyPDMat} <: Kernel
    Σ::T
end
@inline (k::EmpiricalKernel)(q::Int, q′::Int) = k.Σ[q, q′]
@inline (k::EmpiricalKernel)(q::Int) = k.Σ[q, q]

_pairwise(k::EmpiricalKernel, X::AV) = X == eachindex(k) ? k.Σ : k.Σ[X, X]

function _pairwise(k::EmpiricalKernel, X::AV, X′::AV)
    return X == eachindex(k) && X′ == eachindex(k) ? k.Σ : k.Σ[X, X′]
end
AbstractMatrix(k::EmpiricalKernel) = k.Σ

# +(x::ZeroKernel, x′::ZeroKernel) = zero(x)
# function +(k::CrossKernel, k′::CrossKernel)
#     @assert size(k) == size(k′)
#     if iszero(k)
#         return k′
#     elseif iszero(k′)
#         return k
#     else
#         return CompositeCrossKernel(+, k, k′)
#     end
# end
# function +(k::Kernel, k′::Kernel)
#     @assert size(k) == size(k′)
#     if iszero(k)
#         return k′
#     elseif iszero(k′)
#         return k
#     else
#         return CompositeKernel(+, k, k′)
#     end
# end
# function *(k::Kernel, k′::Kernel)
#     @assert size(k) == size(k′)
#     return iszero(k) || iszero(k′) ? zero(k) : CompositeKernel(*, k, k′)
# end
# function *(k::CrossKernel, k′::CrossKernel)
#     @assert size(k) == size(k′)
#     return iszero(k) || iszero(k′) ? zero(k) : CompositeCrossKernel(*, k, k′)
# end
