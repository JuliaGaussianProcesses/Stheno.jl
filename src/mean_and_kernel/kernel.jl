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
ew(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods
ew(::Exp, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::Exp, x::AV{<:Real}) = pw(k, x, x)
pw(k::Exp, x::StepRangeLen{<:Real}) = toep_pw(k, x)


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
