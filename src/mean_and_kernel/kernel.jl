using LinearAlgebra

import LinearAlgebra: AbstractMatrix, AdjOrTransAbsVec, AdjointAbsVec
import Base: +, *, ==, size, eachindex, print, eltype, zero
import Distances: pairwise, colwise, sqeuclidean, SqEuclidean
import Base.Broadcast: broadcast_shape


############################# Define CrossKernels and Kernels ##############################

abstract type CrossKernel end
abstract type Kernel <: CrossKernel end



################################ Util. for Toeplitz matrices ###############################

function toep_pw(k::CrossKernel, x::StepRangeLen{T}, x′::StepRangeLen{V}) where {T, V}
    if x.step == x′.step
        return Toeplitz(ew(k, x, fill(x′[1], length(x))), ew(k, fill(x[1], length(x′)), x′))
    else
        return pw(k, collect(x), collect(x′))
    end
end

toep_pw(k::Kernel, x::StepRangeLen) = SymmetricToeplitz(ew(k, x, fill(x[1], length(x))))

function toep_map(k::Kernel, x::StepRangeLen{T}, x′::StepRangeLen{V}) where {T, V}
    if x.step == x′.step
        return fill(
            ew(k, collect(x[1:1]), collect(x′[1:1]))[1],
            broadcast_shape(size(x), size(x′)),
        )
    else
        return ew(k, collect(x), collect(x′))
    end
end



################################ Define some basic kernels #################################



"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
ZeroKernel() = ZeroKernel{Float64}()
eltype(::ZeroKernel{T}) where {T} = T
zero(::CrossKernel) = ZeroKernel()

# Binary methods.
ew(k::ZeroKernel, x::AV, x′::AV) = zeros(eltype(k), broadcast_shape(size(x), size(x′)))
pw(k::ZeroKernel, x::AV, x′::AV) = zeros(eltype(k), length(x), length(x′))

# Unary methods.
ew(k::ZeroKernel, x::AV) = zeros(eltype(k), length(x))
pw(k::ZeroKernel, x::AV) = zeros(eltype(k), length(x), length(x))



"""
    OneKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct OneKernel{T<:Real} <: Kernel end
OneKernel() = OneKernel{Float64}()
eltype(k::OneKernel{T}) where {T} = T

# Binary methods.
ew(k::OneKernel, x::AV, x′::AV) = ones(eltype(k), broadcast_shape(size(x), size(x′)))
pw(k::OneKernel, x::AV, x′::AV) = ones(eltype(k), length(x), length(x′))

# Unary methods.
ew(k::OneKernel, x::AV) = ones(eltype(k), length(x))
pw(k::OneKernel, x::AV) = ones(eltype(k), length(x), length(x))



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
ew(k::ConstKernel, x::AV, x′::AV) = fill(k.c, broadcast_shape(size(x), size(x′))...)
pw(k::ConstKernel, x::AV, x′::AV) = fill(k.c, length(x), length(x′))

# Unary methods.
ew(k::ConstKernel, x::AV) = fill(k.c, length(x))
pw(k::ConstKernel, x::AV) = pw(k, x, x)



"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel (no free parameters).
"""
struct EQ <: Kernel end

# Binary methods.
ew(::EQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-sqeuclidean.(x, x′) ./ 2)
pw(::EQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-sqeuclidean.(x, x′') ./ 2)
ew(::EQ, X::ColsAreObs, X′::ColsAreObs) = exp.(.-colwise(SqEuclidean(), X.X, X′.X) ./ 2)
pw(::EQ, X::ColsAreObs, X′::ColsAreObs) = exp.(.-pw(SqEuclidean(), X.X, X′.X; dims=2) ./ 2)

ew(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
ew(::EQ, x::AV) = ones(eltype(x), length(x))
pw(::EQ, x::AV{<:Real}) = pw(EQ(), x, x)
ew(::EQ, X::ColsAreObs) = ones(eltype(X.X), length(X))
pw(::EQ, X::ColsAreObs) = exp.(.-pw(SqEuclidean(), X.X; dims=2) ./ 2)
pw(k::EQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

# Optimised adjoints. These really do count in terms of performance (I think).
@adjoint function ew(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = ew(EQ(), x, x′)
    return s, function(Δ)
        x̄′ = (x .- x′) .* Δ .* s
        return nothing, -x̄′, x̄′
    end
end
@adjoint function pw(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = pw(EQ(), x, x′)
    return s, function(Δ)
        x̄′ = Δ .* (x .- x′') .* s
        return nothing, -reshape(sum(x̄′; dims=2), :), reshape(sum(x̄′; dims=1), :)
    end
end
@adjoint function pw(::EQ, x::AV{<:Real})
    s = pw(EQ(), x)
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
ew(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-2 .* sin.(π .* abs.(x .- x′)).^2)
pw(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-2 .* sin.(π .* abs.(x .- x′')).^2)
ew(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
ew(::PerEQ, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::PerEQ, x::AV{<:Real}) = pw(k, x, x)
pw(k::PerEQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)



"""
    Exp <: Kernel

The standardised Exponential kernel.
"""
struct Exp <: Kernel end

# Binary methods
ew(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-abs.(x .- x′))
pw(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-abs.(x .- x′'))
ew(k::Exp, x::ColsAreObs, x′::ColsAreObs) = exp.(.-colwise(Euclidean(), x.X, x′.X))
pw(k::Exp, x::ColsAreObs, x′::ColsAreObs) = exp.(.-pairwise(Euclidean(), x.X, x′.X; dims=2))
ew(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods
ew(::Exp, x::AV{<:Real}) = ones(eltype(x), length(x))
ew(::Exp, x::ColsAreObs{T}) where {T} = ones(T, length(x))
pw(k::Exp, x::AV{<:Real}) = pw(k, x, x)
pw(k::Exp, x::ColsAreObs) = exp.(.-pairwise(Euclidean(), x.X; dims=2))
pw(k::Exp, x::StepRangeLen{<:Real}) = toep_pw(k, x)



"""
    Matern12

Equivalent to the Exponential kernel.
"""
const Matern12 = Exp



"""
    Matern32 <: Kernel

The Matern kernel with ν = 3 / 2
"""
struct Matern32 <: Kernel end

function _matern32(d)
    d = sqrt(3) * d
    return (1 + d) * exp(-d)
end

# Binary methods
ew(::Matern32, x::AV{<:Real}, x′::AV{<:Real}) = _matern32.(abs.(x .- x′))
pw(::Matern32, x::AV{<:Real}, x′::AV{<:Real}) = _matern32.(abs.(x .- x′'))
ew(k::Matern32, x::ColsAreObs, x′::ColsAreObs) = _matern32.(colwise(Euclidean(), x.X, x′.X))
function pw(k::Matern32, x::ColsAreObs, x′::ColsAreObs)
    return _matern32.(pairwise(Euclidean(), x.X, x′.X; dims=2))
end

# Unary methods
ew(::Matern32, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::Matern32, x::AV{<:Real}) = pw(k, x, x)
ew(::Matern32, x::ColsAreObs{T}) where {T<:Real} = ones(T, length(x))
pw(::Matern32, x::ColsAreObs) = _matern32.(pairwise(Euclidean(), x.X; dims=2))



"""
    Matern52 <: Kernel

The Matern kernel with ν = 5 / 2
"""
struct Matern52 <: Kernel end

function _matern52(d)
    d = sqrt(5) * d
    return (1 + d + d^2 / 3) * exp(-d)
end

# Binary methods
ew(::Matern52, x::AV{<:Real}, x′::AV{<:Real}) = _matern52.(abs.(x .- x′))
pw(::Matern52, x::AV{<:Real}, x′::AV{<:Real}) = _matern52.(abs.(x .- x′'))
ew(k::Matern52, x::ColsAreObs, x′::ColsAreObs) = _matern52.(colwise(Euclidean(), x.X, x′.X))
function pw(k::Matern52, x::ColsAreObs, x′::ColsAreObs)
    return _matern52.(pairwise(Euclidean(), x.X, x′.X; dims=2))
end

# Unary methods
ew(::Matern52, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::Matern52, x::AV{<:Real}) = pw(k, x, x)
ew(::Matern52, x::ColsAreObs{T}) where {T<:Real} = ones(T, length(x))
pw(::Matern52, x::ColsAreObs) = _matern52.(pairwise(Euclidean(), x.X; dims=2))



"""
    RQ <: Kernel

The standardised Rational Quadratic, with kurtosis `α`.
"""
struct RQ{Tα<:Real} <: Kernel
    α::Tα
end

_rq(d, α) = (1 + d / (2α))^(-α)

# Binary methods
ew(k::RQ, x::AV{<:Real}, x′::AV{<:Real}) = _rq.(sqeuclidean.(x, x′), k.α)
pw(k::RQ, x::AV{<:Real}, x′::AV{<:Real}) = _rq.(sqeuclidean.(x, x′'), k.α)
function ew(k::RQ, x::ColsAreObs, x′::ColsAreObs)
    return _rq.(colwise(SqEuclidean(), x.X, x′.X), k.α)
end
function pw(k::RQ, x::ColsAreObs, x′::ColsAreObs)
    return _rq.(pairwise(SqEuclidean(), x.X, x′.X; dims=2), k.α)
end

# Unary methods
ew(k::RQ, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::RQ, x::AV{<:Real}) = pw(k, x, x)
ew(k::RQ, x::ColsAreObs{T}) where {T<:Real} = ones(T, length(x))
pw(k::RQ, x::ColsAreObs) = _rq.(pairwise(SqEuclidean(), x.X; dims=2), k.α)



"""
    Linear{T<:Real} <: Kernel

The standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

# Binary methods
ew(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′
pw(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′'
ew(k::Linear, x::ColsAreObs, x′::ColsAreObs) = reshape(sum(x.X .* x′.X; dims=1), :)
pw(k::Linear, x::ColsAreObs, x′::ColsAreObs) = x.X' * x′.X

# Unary methods
ew(k::Linear, x::AV{<:Real}) = x.^2
pw(k::Linear, x::AV{<:Real}) = x .* x'
ew(k::Linear, x::ColsAreObs) = reshape(sum(abs2.(x.X); dims=1), :)
pw(k::Linear, x::ColsAreObs) = x.X' * x.X



"""
    Noise{T<:Real} <: Kernel

The standardised aleatoric white-noise kernel. Isn't really a kernel, but never mind...
"""
struct Noise{T<:Real} <: Kernel end
Noise() = Noise{Int}()
eltype(k::Noise{T}) where {T} = T

# Binary methods.
ew(k::Noise, x::AV, x′::AV) = zeros(eltype(k), broadcast_shape(size(x), size(x′))...)
pw(k::Noise, x::AV, x′::AV) = zeros(eltype(k), length(x), length(x′))

# Unary methods.
ew(k::Noise, x::AV) = ones(eltype(k), length(x))
pw(k::Noise, x::AV) = diagm(0=>ones(eltype(k), length(x)))



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
