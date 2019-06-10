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

@adjoint function map(k::CrossKernel, x::AV)
    return Zygote.forward((k, x)->materialize(_map(k, x)), k, x)
end
@adjoint function map(k::CrossKernel, x::GPUVector)
    return Zygote.forward((k, x)->materialize(_map(k, x)), k, x)
end

"""
    map(k::CrossKernel, x::AV, x′::AV)

map `k` over the elements of `x` and `x′`.
"""
map(k::CrossKernel, x::AV, x′::AV) = materialize(_map(k, x, x′))
map(k::CrossKernel, x::GPUVector, x′::GPUVector) = materialize(_map(k, x, x′))

@adjoint function map(k::CrossKernel, x::AV, x′::AV)
    return Zygote.forward((k, x, x′)->materialize(_map(k, x, x′)), k, x, x′)
end
@adjoint function map(k::CrossKernel, x::GPUVector, x′::GPUVector)
    return Zygote.forward((k, x, x′)->materialize(_map(k, x, x′)), k, x, x′)
end

"""
    pairwise(f, x::AV)

Compute the `length(x) × length(x′)` matrix whose `(p, q)`th element is `k(x[p], x[q])`.
`_pw` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV) = materialize(_pw(k, x))
@adjoint function pairwise(k::CrossKernel, x::AV)
    return Zygote.forward((k, x)->materialize(_pw(k, x)), k, x)
end

"""
    pairwise(f, x::AV, x′::AV)

Compute the `length(x) × length(x′)` matrix whose `(p, q)`th element is `k(x[p], x′[q])`.
`_pw` is called and `materialize`d, meaning that operations can be fused using Julia's
broadcasting machinery if required.
"""
pairwise(k::CrossKernel, x::AV, x′::AV) = materialize(_pw(k, x, x′))
@adjoint function pairwise(k::CrossKernel, x::AV, x′::AV)
    return Zygote.forward((k, x, x′)->materialize(_pw(k, x, x′)), k, x, x′)
end


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
ZeroKernel() = ZeroKernel{Float64}()
eltype(::ZeroKernel{T}) where {T} = T
zero(::CrossKernel) = ZeroKernel()

# Binary methods.
map(k::ZeroKernel, x::AV, x′::AV) = zeros(eltype(k), broadcast_shape(size(x), size(x′))...)
pw(k::ZeroKernel, x::AV, x′::AV) = zeros(eltype(k), length(x), length(x′))

# Unary methods.
map(k::ZeroKernel, x::AV) = zeros(eltype(k), length(x))
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
map(k::OneKernel, x::AV, x′::AV) = ones(eltype(k), broadcast_shape(size(x), size(x′))...)
pw(k::OneKernel, x::AV, x′::AV) = ones(eltype(k), length(x), length(x′))

# Unary methods.
map(k::OneKernel, x::AV) = ones(eltype(k), length(x))
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
map(k::ConstKernel, x::AV, x′::AV) = fill(k.c, broadcast_shape(size(x), size(x′))...)
pw(k::ConstKernel, x::AV, x′::AV) = fill(k.c, length(x), length(x′))

# Unary methods.
map(k::ConstKernel, x::AV) = fill(k.c, length(x))
pw(k::ConstKernel, x::AV) = pw(k, x, x)


"""
    EQ <: Kernel

The standardised Exponentiated Quadratic kernel (no free parameters).
"""
struct EQ <: Kernel end

# Binary methods.
map(::EQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-sqeuclidean.(x, x′) ./ 2)
pw(::EQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-sqeuclidean.(x, x′') ./ 2)
map(::EQ, X::ColsAreObs, X′::ColsAreObs) = exp.(.-colwise(SqEuclidean(), X.X, X′.X) ./ 2)
pw(::EQ, X::ColsAreObs, X′::ColsAreObs) = exp.(.-pw(SqEuclidean(), X.X, X′.X; dims=2) ./ 2)

map(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::EQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
map(::EQ, x::AV) = ones(eltype(x), length(x))
pw(::EQ, x::AV{<:Real}) = _pw(EQ(), x, x)
map(::EQ, X::ColsAreObs) = ones(eltype(X.X), length(X))
pw(::EQ, X::ColsAreObs) = bcd(x->exp(-x / 2), pw(SqEuclidean(), X.X; dims=2))
pw(k::EQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)

# Optimised adjoints. These really do count in terms of performance (I think).
@adjoint function map(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = materialize(_map(EQ(), x, x′))
    return s, function(Δ)
        x̄′ = (x .- x′) .* Δ .* s
        return nothing, -x̄′, x̄′
    end
end
@adjoint function pw(::EQ, x::AV{<:Real}, x′::AV{<:Real})
    s = materialize(_pw(EQ(), x, x′))
    return s, function(Δ)
        x̄′ = Δ .* (x .- x′') .* s
        return nothing, -reshape(sum(x̄′; dims=2), :), reshape(sum(x̄′; dims=1), :)
    end
end
@adjoint function pw(::EQ, x::AV{<:Real})
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
map(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-2 .* sin.(π .* abs.(x .- x′)).^2)
pw(k::PerEQ, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-2 .* sin.(π .* abs.(x .- x′')).^2)
map(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::PerEQ, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods.
map(::PerEQ, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::PerEQ, x::AV{<:Real}) = pw(k, x, x)
pw(k::PerEQ, x::StepRangeLen{<:Real}) = toep_pw(k, x)


"""
    Exp <: Kernel

The standardised Exponential kernel.
"""
struct Exp <: Kernel end

# Binary methods
map(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-abs.(x .- x′))
pw(k::Exp, x::AV{<:Real}, x′::AV{<:Real}) = exp.(.-abs.(x .- x′'))
map(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_map(k, x, x′)
pw(k::Exp, x::StepRangeLen{<:Real}, x′::StepRangeLen{<:Real}) = toep_pw(k, x, x′)

# Unary methods
map(::Exp, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::Exp, x::AV{<:Real}) = pw(k, x, x)
pw(k::Exp, x::StepRangeLen{<:Real}) = toep_pw(k, x)


"""
    Linear{T<:Real} <: Kernel

The standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

# Binary methods
map(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′
pw(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′'
map(k::Linear, x::ColsAreObs, x′::ColsAreObs) = reshape(sum(x.X .* x′.X; dims=1), :)
pw(k::Linear, x::ColsAreObs, x′::ColsAreObs) = x.X' * x′.X

# Unary methods
map(k::Linear, x::AV{<:Real}) = x.^2
pw(k::Linear, x::AV{<:Real}) = x .* x'
map(k::Linear, x::ColsAreObs) = reshape(sum(abs2.(x.X); dims=1), :)
pw(k::Linear, x::ColsAreObs) = x.X' * x.X


"""
    Noise{T<:Real} <: Kernel

The standardised aleatoric white-noise kernel. Isn't really a kernel, but never mind...
"""
struct Noise{T<:Real} <: Kernel end
Noise() = Noise{Int}()
eltype(k::Noise{T}) where {T} = T

# Binary methods.
map(k::Noise, x::AV, x′::AV) = zeros(eltype(k), broadcast_shape(size(x), size(x′))...)
pw(k::Noise, x::AV, x′::AV) = zeros(eltype(k), length(x), length(x′))

# Unary methods.
map(k::Noise, x::AV) = ones(eltype(k), length(x))
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
