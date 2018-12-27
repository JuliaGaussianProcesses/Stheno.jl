using LinearAlgebra, GPUArrays
using Base.Broadcast: DefaultArrayStyle
using GPUArrays: GPUVector

import LinearAlgebra: AbstractMatrix, AdjOrTransAbsVec, AdjointAbsVec
import Base: +, *, ==, size, eachindex, print, eltype
import Distances: pairwise, colwise, sqeuclidean, SqEuclidean
import Base.Broadcast: broadcasted, materialize, broadcast_shape

export CrossKernel, Kernel, cov, xcov, EQ, PerEQ, RQ, Linear, Poly, Noise, Wiener,
    WienerVelocity, Exp, ConstantKernel, isstationary, ZeroKernel, OneKernel, pairwise



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
`_pw` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV) = materialize(_pw(k, x))

"""
    pairwise(f, x::AV, x′::AV)

Compute the `length(x) × length(x′)` matrix whose `(p, q)`th element is `k(x[p], x′[q])`.
`_pw` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV, x′::AV) = materialize(_pw(k, x, x′))



################################ Util. for Toeplitz matrices ###############################

function toep_pw(k::CrossKernel, x::StepRangeLen, x′::StepRangeLen)
    if x.step == x′.step
        return Toeplitz(
            map(k, x, Fill(x′[1], length(x))),
            map(k, Fill(x[1], length(x′)), x′),
        )
    else
        return invoke(_pw, Tuple{typeof(k), AV, AV}, k, x, x′)
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
ZeroKernel() = ZeroKernel{Int}()
eltype(::ZeroKernel{T}) where {T} = T

# Binary methods.
(k::ZeroKernel)(x, x′) = zero(eltype(k))
_map(k::ZeroKernel, x::AV, x′::AV) = Zeros{eltype(k)}(broadcast_shape(size(x), size(x′)))
_pw(k::ZeroKernel, x::AV, x′::AV) = Zeros{eltype(k)}(length(x), length(x′))
@adjoint (k::ZeroKernel)(x, x′) = k(x, x′), Δ->(zero(x), zero(x′))
@adjoint function _map(k::ZeroKernel, x::AV, x′::AV)
    return _map(k, x, x′), Δ->(nothing, Zeros{Int}(size(x)), Zeros{Int}(size(x′)))
end
@adjoint function _pw(k::ZeroKernel, x::AV, x′::AV)
    return _pw(k, x, x′), Δ->(nothing, Zeros{Int}(size(x)), Zeros{Int}(size(x′)))
end

# Unary methods.
(k::ZeroKernel)(x) = zero(eltype(k))
_map(k::ZeroKernel, x::AV) = Zeros{eltype(k)}(length(x))
_pw(k::ZeroKernel, x::AV) = Zeros{eltype(k)}(length(x), length(x))
@adjoint (k::ZeroKernel)(x) = k(x), Δ->(zero(x),)
@adjoint _map(k::ZeroKernel, x::AV) = _map(k, x), Δ->(nothing, Zeros{Int}(size(x)))
@adjoint _pw(k::ZeroKernel, x::AV) = _pw(k, x), Δ->(nothing, Zeros{Int}(size(x)))


"""
    OneKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct OneKernel{T<:Real} <: Kernel end
OneKernel() = OneKernel{Int}()
eltype(k::OneKernel{T}) where {T} = T

# Binary methods.
(k::OneKernel)(x, x′) = one(eltype(k))
_map(k::OneKernel, x::AV, x′::AV) = Ones{eltype(k)}(broadcast_shape(size(x), size(x′)))
_pw(k::OneKernel, x::AV, x′::AV) = Ones{eltype(k)}(length(x), length(x′))
@adjoint (k::OneKernel)(x, x′) = k(x, x′), Δ->(zero(x), zero(x′))
@adjoint function _map(k::OneKernel, x::AV, x′::AV)
    return _map(k, x, x′), Δ->(nothing, Zeros{Int}(size(x)), Zeros{Int}(size(x′)))
end
@adjoint function _pw(k::OneKernel, x::AV, x′::AV)
    return _pw(k, x, x′), Δ->(nothing, Zeros{Int}(size(x)), Zeros{Int}(size(x′)))
end

# Unary methods.
(k::OneKernel)(x) = one(eltype(k))
_map(k::OneKernel, x::AV) = Ones{eltype(k)}(length(x))
_pw(k::OneKernel, x::AV) = Ones{eltype(k)}(length(x), length(x))
@adjoint (k::OneKernel)(x) = k(x), Δ->(zero(x),)
@adjoint _map(k::OneKernel, x::AV) = _map(k, x), Δ->(nothing, Zeros{Int}(size(x)))
@adjoint _pw(k::OneKernel, x::AV) = _pw(k, x), Δ->(nothing, Zeros{Int}(size(x)))


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
function _pw(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd((x, x′)->-sqeuclidean(x, x′) / 2, x, x′'))
end
function _map(::EQ, X::ColsAreObs, X′::ColsAreObs)
    return bcd(x->exp(-x / 2), colwise(SqEuclidean(), X.X, X′.X))
end
function _pw(::EQ, X::ColsAreObs, X′::ColsAreObs)
    return bcd(x->exp(-x / 2), pairwise(SqEuclidean(), X.X, X′.X))
end
_map(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pw(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
(::EQ)(x::Real) = one(x)
(::EQ)(x::AV{<:Real}) = one(eltype(x))
_map(::EQ, x::AV) = Ones{eltype(x)}(length(x))
_map(::EQ, X::ColsAreObs) = Ones{eltype(X.X)}(length(X))
_pw(::EQ, x::AV{<:Real}) = _pw(EQ(), x, x)
_pw(::EQ, X::ColsAreObs) = bcd(x->exp(-x / 2), pairwise(SqEuclidean(), X.X))
_pw(::EQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

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
@adjoint function _pw(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = materialize(_pw(EQ(), x, x′))
    return s, function(Δ)
        x̄′ = Δ .* (x .- x′') .* s
        return nothing, -reshape(sum(x̄′; dims=2), :), reshape(sum(x̄′; dims=1), :)
    end
end
@adjoint function _pw(::EQ, x::AV{<:Real})
    s = materialize(_pw(EQ(), x))
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
function _pw(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd(x->-2x^2, bcd(sin, bcd((x, x′)->π * abs(x - x′), x, x′'))))
end
_map(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pw(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
(::PerEQ)(x::Real) = one(typeof(x))
_map(::PerEQ, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pw(k::PerEQ, x::AV{<:Real}) = _pw(k, x, x)
_pw(::PerEQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

@adjoint (k::PerEQ)(x::Real) = (k(x), _->(zero(typeof(x)),))
@adjoint function _map(k::PerEQ, x::AV{<:Real})
    return _map(k, x), _->(nothing, Zeros{eltype(x)}(length(x),))
end


"""
    Exp <: Kernel

The standardised Exponential kernel.
"""
struct Exp <: Kernel end

# Binary methods
(::Exp)(x::Real, x′::Real) = exp(-abs(x - x′))
_map(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′))
_pw(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′'))

# Unary methods
(::Exp)(x) = 1
_map(::Exp, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pw(k::Exp, x::AV{<:Real}) = _pw(k, x, x)

@adjoint _map(k::Exp, x::AV{<:Real}) = _map(k, x), Δ->(nothing, Zeros{eltype(x)}(length(x)))


"""
    Linear{T<:Real} <: Kernel

Standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

# Binary methods
(k::Linear)(x, x′) = dot(x, x′)
_map(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′
_pw(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′'
_map(k::Linear, x::ColsAreObs, x′::ColsAreObs) = reshape(sum(x.X .* x′.X; dims=1), :)
_pw(k::Linear, x::ColsAreObs, x′::ColsAreObs) = x.X' * x′.X

# Unary methods
(k::Linear)(x) = dot(x, x)
_map(k::Linear, x::AV{<:Real}) = x.^2
_pw(k::Linear, x::AV{<:Real}) = x .* x'
_map(k::Linear, x::ColsAreObs) = reshape(sum(abs2, x.X; dims=1), :)
_pw(k::Linear, x::ColsAreObs) = x.X' * x.X


"""
    Noise{T<:Real} <: Kernel

Standardised aleatoric white-noise kernel.
"""
struct Noise{T<:Real} <: Kernel end
Noise() = Noise{Int}()
eltype(k::Noise{T}) where {T} = T

# Binary methods.
(k::Noise)(x, x′) = zero(eltype(k))
_map(k::Noise, x::AV, x′::AV) = Zeros{eltype(k)}(broadcast_shape(size(x), size(x′)))
_pw(k::Noise, x::AV, x′::AV) = Zeros{eltype(k)}(length(x), length(x′))

@adjoint function _map(k::Noise{T}, x::AV, x′::AV) where {T}
    return _map(k, x, x′), Δ->(nothing, Zeros{T}(length(x)), Zeros{T}(length(x′)))
end
@adjoint function _pw(k::Noise{T}, x::AV, x′::AV) where {T}
    return _pw(k, x, x′), Δ->(nothing, Zeros{T}(length(x)), Zeros{T}(length(x′)))
end

# Unary methods.
(k::Noise)(x) = one(eltype(k))
_map(k::Noise, x::AV) = Ones{eltype(k)}(length(x))
_pw(k::Noise, x::AV) = Diagonal(Ones{eltype(k)}(length(x)))

@adjoint _map(k::Noise, x::AV) = _map(k, x), Δ->(nothing, Zeros{eltype(k)}(length(x)))
@adjoint _pw(k::Noise, x::AV) = _pw(k, x), Δ->(nothing, Zeros{eltype(k)}(length(x)))

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

_pw(k::EmpiricalKernel, X::AV) = X == eachindex(k) ? k.Σ : k.Σ[X, X]

function _pw(k::EmpiricalKernel, X::AV, X′::AV)
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
