using LinearAlgebra, GPUArrays
using Base.Broadcast: DefaultArrayStyle
using GPUArrays: GPUVector

import LinearAlgebra: AbstractMatrix, AdjOrTransAbsVec, AdjointAbsVec
import Base: +, *, ==, size, eachindex, print, eltype, zero
import Distances: pairwise, colwise, sqeuclidean, SqEuclidean
import Base.Broadcast: broadcast_shape


############################# Define CrossKernels and Kernels ##############################

abstract type CrossKernel end
abstract type Kernel <: CrossKernel end

"""
    map(k::CrossKernel, x::AV)

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

function toep_pw(k::CrossKernel, x::StepRangeLen{T}, x′::StepRangeLen{V}) where {T, V}
    if x.step == x′.step
        return Toeplitz(
            map(k, x, Fill(x′[1], length(x))),
            map(k, Fill(x[1], length(x′)), x′),
        )
    else
        return _pw(k, collect(x), collect(x′))
    end
end

toep_pw(k::Kernel, x::StepRangeLen) = SymmetricToeplitz(map(k, x, Fill(x[1], length(x))))

function toep_map(k::Kernel, x::StepRangeLen{T}, x′::StepRangeLen{V}) where {T, V}
    if x.step == x′.step
        return Fill(
            map(k, collect(x[1:1]), collect(x′[1:1]))[1],
            broadcast_shape(size(x), size(x′)),
        )
    else
        return _map(k, collect(x), collect(x′))
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
zero(::CrossKernel) = ZeroKernel()

# Binary methods.
_map(k::ZeroKernel, x::AV, x′::AV) = Zeros{eltype(k)}(broadcast_shape(size(x), size(x′))...)
_pw(k::ZeroKernel, x::AV, x′::AV) = Zeros{eltype(k)}(length(x), length(x′))

# Unary methods.
_map(k::ZeroKernel, x::AV) = Zeros{eltype(k)}(length(x))
_pw(k::ZeroKernel, x::AV) = Zeros{eltype(k)}(length(x), length(x))


"""
    OneKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct OneKernel{T<:Real} <: Kernel end
OneKernel() = OneKernel{Int}()
eltype(k::OneKernel{T}) where {T} = T

# Binary methods.
_map(k::OneKernel, x::AV, x′::AV) = Ones{eltype(k)}(broadcast_shape(size(x), size(x′))...)
_pw(k::OneKernel, x::AV, x′::AV) = Ones{eltype(k)}(length(x), length(x′))

# Unary methods.
_map(k::OneKernel, x::AV) = Ones{eltype(k)}(length(x))
_pw(k::OneKernel, x::AV) = Ones{eltype(k)}(length(x), length(x))


"""
    ConstKernel{T} <: Kernel

A rank 1 kernel that returns the same value `c` everywhere.
"""
struct ConstKernel{T} <: Kernel
    c::T
end

# A hack to make this work with Zygote, which can't handle parametrised function calls.
const_kernel(c, x, x′) = c

# Binary methods.
_map(k::ConstKernel, x::AV, x′::AV) = Fill(k.c, broadcast_shape(size(x), size(x′))...)
_pw(k::ConstKernel, x::AV, x′::AV) = Fill(k.c, length(x), length(x′))

# Unary methods.
_map(k::ConstKernel, x::AV) = Fill(k.c, length(x))
_pw(k::ConstKernel, x::AV) = _pw(k, x, x)


"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel (no free parameters).
"""
struct EQ <: Kernel end

# Binary methods.
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
_map(::EQ, x::AV) = Ones{eltype(x)}(length(x))
_pw(::EQ, x::AV{<:Real}) = _pw(EQ(), x, x)
_map(::EQ, X::ColsAreObs) = Ones{eltype(X.X)}(length(X))
_pw(::EQ, X::ColsAreObs) = bcd(x->exp(-x / 2), pairwise(SqEuclidean(), X.X))
_pw(k::EQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

# Optimised adjoints. These really do count in terms of performance (I think).
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


"""
    PerEQ

The usual periodic kernel derived by mapping the input domain onto the unit circle.
"""
struct PerEQ <: Kernel end

# Binary methods.
function _map(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd(x->-2x^2, bcd(sin, bcd((x, x′)->π * abs(x - x′), x, x′))))
end
function _pw(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real})
    return bcd(exp, bcd(x->-2x^2, bcd(sin, bcd((x, x′)->π * abs(x - x′), x, x′'))))
end
_map(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pw(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
_map(::PerEQ, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pw(k::PerEQ, x::AV{<:Real}) = _pw(k, x, x)
_pw(k::PerEQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)


"""
    Exp <: Kernel

The standardised Exponential kernel.
"""
struct Exp <: Kernel end

# Binary methods
_map(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′))
_pw(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = bcd(exp, bcd((x, x′)->-abs(x - x′), x, x′'))
_map(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
_pw(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods
_map(::Exp, x::AV{<:Real}) = Ones{eltype(x)}(length(x))
_pw(k::Exp, x::AV{<:Real}) = _pw(k, x, x)
_pw(k::Exp, x::StepRangeLen{<:Real}) = toep_pw(k, x)


"""
    Linear{T<:Real} <: Kernel

The standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

# Binary methods
_map(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′
_pw(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′'
_map(k::Linear, x::ColsAreObs, x′::ColsAreObs) = reshape(sum(x.X .* x′.X; dims=1), :)
_pw(k::Linear, x::ColsAreObs, x′::ColsAreObs) = x.X' * x′.X

# Unary methods
_map(k::Linear, x::AV{<:Real}) = x.^2
_pw(k::Linear, x::AV{<:Real}) = x .* x'
_map(k::Linear, x::ColsAreObs) = reshape(sum(abs2.(x.X); dims=1), :)
_pw(k::Linear, x::ColsAreObs) = x.X' * x.X


"""
    Noise{T<:Real} <: Kernel

The standardised aleatoric white-noise kernel. Isn't really a kernel, but never mind...
"""
struct Noise{T<:Real} <: Kernel end
Noise() = Noise{Int}()
eltype(k::Noise{T}) where {T} = T

# Binary methods.
_map(k::Noise, x::AV, x′::AV) = Zeros{eltype(k)}(broadcast_shape(size(x), size(x′))...)
_pw(k::Noise, x::AV, x′::AV) = Zeros{eltype(k)}(length(x), length(x′))

# Unary methods.
_map(k::Noise, x::AV) = Ones{eltype(k)}(length(x))
_pw(k::Noise, x::AV) = Diagonal(Ones{eltype(k)}(length(x)))


"""
    FiniteRank <: Kernel

`k(x, x′) = ϕ(x)' * Σ * ϕ(x′)` where `Σ` is an `M × M` positive-definite matrix given in
terms of its Cholesky factorisation `C`, and `ϕ` returns an `M`-dimensional vector.
"""
struct FiniteRank{TC<:Cholesky, Tϕ} <: Kernel
    C::TC
    ϕ::Tϕ
end
FiniteRank(Σ::Union{Real, AbstractMatrix}, ϕ) = FiniteRank(cholesky(Σ), ϕ)

# Binary methods.
_map(k::FiniteRank, x::AV, x′::AV) = diag_Xt_A_Y(k.ϕ.(x)', k.C, k.ϕ.(x′)')
_pw(k::FiniteRank, x::AV, x′::AV) = Xt_A_Y(k.ϕ.(x)', k.C, k.ϕ.(x′)')

# Unary methods.
_map(k::FiniteRank, x::AV) = diag_Xt_A_X(k.C, k.ϕ.(x)')
_pw(k::FiniteRank, x::AV) = Xt_A_X(k.C, k.ϕ.(x)')


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
