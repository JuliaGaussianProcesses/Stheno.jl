import Base: +, *, zero
import Distances: pairwise, colwise
using Distances: sqeuclidean, SqEuclidean, Euclidean
using Base.Broadcast: broadcast_shape
import Base: exp

abstract type Kernel end

"""
    ZeroKernel <: Kernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
ZeroKernel() = ZeroKernel{Float64}()
zero(::Kernel) = ZeroKernel()

# Binary methods.
ew(k::ZeroKernel{T}, x::AV, x′::AV) where {T} = zeros(T, broadcast_shape(size(x), size(x′)))
pw(k::ZeroKernel{T}, x::AV, x′::AV) where {T} = zeros(T, length(x), length(x′))

# Unary methods.
ew(k::ZeroKernel{T}, x::AV) where {T} = zeros(T, length(x))
pw(k::ZeroKernel{T}, x::AV) where {T} = zeros(T, length(x), length(x))



"""
    OneKernel{T<:Real} <: Kernel

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct OneKernel{T<:Real} <: Kernel end
OneKernel() = OneKernel{Float64}()

# Binary methods.
ew(k::OneKernel{T}, x::AV, x′::AV) where {T} = ones(T, broadcast_shape(size(x), size(x′)))
pw(k::OneKernel{T}, x::AV, x′::AV) where {T} = ones(T, length(x), length(x′))

# Unary methods.
ew(k::OneKernel{T}, x::AV) where {T} = ones(T, length(x))
pw(k::OneKernel{T}, x::AV) where {T} = ones(T, length(x), length(x))



"""
    ConstKernel{T} <: Kernel

A rank 1 kernel that returns the same value `c` everywhere.
"""
struct ConstKernel{T} <: Kernel
    c::T
end

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

# Unary methods.
ew(::EQ, x::AV) = ones(eltype(x), length(x))
pw(::EQ, x::AV{<:Real}) = pw(EQ(), x, x)
ew(::EQ, X::ColsAreObs) = ones(eltype(X.X), length(X))
pw(::EQ, X::ColsAreObs) = exp.(.-pw(SqEuclidean(), X.X; dims=2) ./ 2)

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

# Unary methods.
ew(::PerEQ, x::AV{<:Real}) = ones(eltype(x), length(x))
pw(k::PerEQ, x::AV{<:Real}) = pw(k, x, x)



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

# Unary methods
ew(::Exp, x::AV{<:Real}) = ones(eltype(x), length(x))
ew(::Exp, x::ColsAreObs{T}) where {T} = ones(T, length(x))
pw(k::Exp, x::AV{<:Real}) = pw(k, x, x)
pw(k::Exp, x::ColsAreObs) = exp.(.-pairwise(Euclidean(), x.X; dims=2))



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

# Binary methods.
ew(k::Noise{T}, x::AV, x′::AV) where {T} = zeros(T, broadcast_shape(size(x), size(x′))...)
pw(k::Noise{T}, x::AV, x′::AV) where {T} = zeros(T, length(x), length(x′))

# Unary methods.
ew(k::Noise{T}, x::AV) where {T} = ones(T, length(x))
pw(k::Noise{T}, x::AV) where {T} = diagm(0=>ones(T, length(x)))



"""
    Sum{Tkl<:Kernel, Tkr<:Kernel} <: Kernel

Represents the sum of two kernels `kl` and `kr` s.t. `k(x, x′) = kl(x, x′) + kr(x, x′)`.
"""
struct Sum{Tkl<:Kernel, Tkr<:Kernel} <: Kernel
    kl::Tkl
    kr::Tkr
end

+(kl::Kernel, kr::Kernel) = Sum(kl, kr)

# Binary methods
ew(k::Sum, x::AV, x′::AV) = ew(k.kl, x, x′) + ew(k.kr, x, x′)
pw(k::Sum, x::AV, x′::AV) = pw(k.kl, x, x′) + pw(k.kr, x, x′)

# Unary methods
ew(k::Sum, x::AV) = ew(k.kl, x) + ew(k.kr, x)
pw(k::Sum, x::AV) = pw(k.kl, x) + pw(k.kr, x)



"""
    Product{Tkl<:Kernel, Tkr<:Kernel} <: Kernel

Represents the product of two kernels `kl` and `kr` s.t. `k(x, x′) = kl(x, x′) kr(x, x′)`.
"""
struct Product{Tkl<:Kernel, Tkr<:Kernel} <: Kernel
    kl::Tkl
    kr::Tkr
end

*(kl::Kernel, kr::Kernel) = Product(kl, kr)

# Binary methods
ew(k::Product, x::AV, x′::AV) = ew(k.kl, x, x′) .* ew(k.kr, x, x′)
pw(k::Product, x::AV, x′::AV) = pw(k.kl, x, x′) .* pw(k.kr, x, x′)

# Unary methods
ew(k::Product, x::AV) = ew(k.kl, x) .* ew(k.kr, x)
pw(k::Product, x::AV) = pw(k.kl, x) .* pw(k.kr, x)



"""
    Scaled{Tσ²<:Real, Tk<:Kernel} <: Kernel

Scale the variance of `Kernel` `k` by `σ²` s.t. `(σ² * k)(x, x′) = σ² * k(x, x′)`.
"""
struct Scaled{Tσ²<:Real, Tk<:Kernel} <: Kernel
    σ²::Tσ²
    k::Tk
end

*(σ²::Real, k::Kernel) = Scaled(σ², k)
*(k::Kernel, σ²) = σ² * k

# Binary methods.
ew(k::Scaled, x::AV, x′::AV) = k.σ² .* ew(k.k, x, x′)
pw(k::Scaled, x::AV, x′::AV) = k.σ² .* pw(k.k, x, x′)

# Unary methods.
ew(k::Scaled, x::AV) = k.σ² .* ew(k.k, x)
pw(k::Scaled, x::AV) = k.σ² .* pw(k.k, x)



"""
    Stretched{Tk<:Kernel} <: Kernel

Apply a length scale to a kernel. Specifically, `k(x, x′) = k(a * x, a * x′)`.
"""
struct Stretched{Ta<:Union{Real, AV{<:Real}, AM{<:Real}}, Tk<:Kernel} <: Kernel
    a::Ta
    k::Tk
end

stretch(k::Kernel, a::Union{Real, AV{<:Real}, AM{<:Real}}) = Stretched(a, k)

# Binary methods (scalar `a`, scalar-valued input)
ew(k::Stretched{<:Real}, x::AV{<:Real}, x′::AV{<:Real}) = ew(k.k, k.a .* x, k.a .* x′)
pw(k::Stretched{<:Real}, x::AV{<:Real}, x′::AV{<:Real}) = pw(k.k, k.a .* x, k.a .* x′)

# Unary methods (scalar)
ew(k::Stretched{<:Real}, x::AV{<:Real}) = ew(k.k, k.a .* x)
pw(k::Stretched{<:Real}, x::AV{<:Real}) = pw(k.k, k.a .* x)

# Binary methods (scalar and vector `a`, vector-valued input)
function ew(k::Stretched{<:Union{Real, AV{<:Real}}}, x::ColsAreObs, x′::ColsAreObs)
    return ew(k.k, ColsAreObs(k.a .* x.X), ColsAreObs(k.a .* x′.X))
end
function pw(k::Stretched{<:Union{Real, AV{<:Real}}}, x::ColsAreObs, x′::ColsAreObs)
    return pw(k.k, ColsAreObs(k.a .* x.X), ColsAreObs(k.a .* x′.X))
end

# Unary methods (scalar and vector `a`, vector-valued input)
ew(k::Stretched{<:Union{Real, AV{<:Real}}}, x::ColsAreObs) = ew(k.k, ColsAreObs(k.a .* x.X))
pw(k::Stretched{<:Union{Real, AV{<:Real}}}, x::ColsAreObs) = pw(k.k, ColsAreObs(k.a .* x.X))

# Binary methods (matrix `a`, vector-valued input)
function ew(k::Stretched{<:AM{<:Real}}, x::ColsAreObs, x′::ColsAreObs)
    return ew(k.k, ColsAreObs(k.a * x.X), ColsAreObs(k.a * x′.X))
end
function pw(k::Stretched{<:AM{<:Real}}, x::ColsAreObs, x′::ColsAreObs)
    return pw(k.k, ColsAreObs(k.a * x.X), ColsAreObs(k.a * x′.X))
end

# Unary methods (scalar and vector `a`, vector-valued input)
ew(k::Stretched{<:AM{<:Real}}, x::ColsAreObs) = ew(k.k, ColsAreObs(k.a * x.X))
pw(k::Stretched{<:AM{<:Real}}, x::ColsAreObs) = pw(k.k, ColsAreObs(k.a * x.X))

# Create convenience versions of each of the kernels that accept a length scale.
for (k, K) in (
    (:eq, :EQ), (:exponential, :Exp), (:matern12, :Matern12), (:matern32, :Matern32),
    (:matern52, :Matern52), (:linear, :Linear),
)
    @eval $k() = $K()
    @eval $k(a::Union{Real, AV{<:Real}, AM{<:Real}}) = stretch($k(), a)
    @eval export $k
end

rq(α) = RQ(α)
rq(α, l) = stretch(rq(α), l)

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
