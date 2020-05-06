import Base: +, *, zero, cos
using Distances: sqeuclidean, SqEuclidean, Euclidean
using Base.Broadcast: broadcast_shape
using LinearAlgebra: isposdef, checksquare

abstract type Kernel end

abstract type BaseKernel <: Kernel end

# API exports
export Kernel, kernel, elementwise, pairwise, ew, pw, stretch, scale

# Kernel exports
export EQ, Exp, PerEQ, Matern12, Matern32, Matern52, RQ, Cosine, Linear, Poly, GammaExp,
    Wiener, WienerVelocity, Precomputed

export scale, stretch

#
# Base Kernels
#

"""
    ZeroKernel <: BaseKernel

A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: BaseKernel end
ZeroKernel() = ZeroKernel{Float64}()
zero(::Kernel) = ZeroKernel()

# Binary methods.
ew(k::ZeroKernel{T}, x::AV, x′::AV) where {T} = zeros(T, broadcast_shape(size(x), size(x′)))
pw(k::ZeroKernel{T}, x::AV, x′::AV) where {T} = zeros(T, length(x), length(x′))

# Unary methods.
ew(k::ZeroKernel{T}, x::AV) where {T} = zeros(T, length(x))
pw(k::ZeroKernel{T}, x::AV) where {T} = zeros(T, length(x), length(x))



"""
    OneKernel{T<:Real} <: BaseKernel

A rank 1 constant `BaseKernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `BaseKernel`.
"""
struct OneKernel{T<:Real} <: BaseKernel end
OneKernel() = OneKernel{Float64}()

# Binary methods.
ew(k::OneKernel{T}, x::AV, x′::AV) where {T} = ones(T, broadcast_shape(size(x), size(x′)))
pw(k::OneKernel{T}, x::AV, x′::AV) where {T} = ones(T, length(x), length(x′))

# Unary methods.
ew(k::OneKernel{T}, x::AV) where {T} = ones(T, length(x))
pw(k::OneKernel{T}, x::AV) where {T} = ones(T, length(x), length(x))



"""
    ConstKernel{T} <: BaseKernel

A rank 1 kernel that returns the same value `c` everywhere.
"""
struct ConstKernel{T} <: BaseKernel
    c::T
end
ConstKernel(c::Real) = ConstKernel(typeof(c)[c])
get_iparam(c::ConstKernel) = c.c

# Binary methods.
ew(k::ConstKernel, x::AV, x′::AV) = fill(k.c[1], broadcast_shape(size(x), size(x′))...)
pw(k::ConstKernel, x::AV, x′::AV) = fill(k.c[1], length(x), length(x′))

# Unary methods.
ew(k::ConstKernel, x::AV) = fill(k.c[1], length(x))
pw(k::ConstKernel, x::AV) = pw(k, x, x)



@doc raw"""
    EQ() <: BaseKernel

The standardised Exponentiated Quadratic kernel. a.k.a. the Radial Basis Function (RBF), or
Squared Exponential kernel.

``k(x, x^\prime) = \exp( -\frac{1}{2} || x - x^\prime||_2^2 )``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct EQ <: BaseKernel end

# Binary methods.
ew(::EQ, x::AV, x′::AV) = exp.(.-ew(SqEuclidean(), x, x′) ./ 2)
pw(::EQ, x::AV, x′::AV) = exp.(.-pw(SqEuclidean(), x, x′) ./ 2)

# Unary methods.
ew(::EQ, x::AV) = exp.(.-ew(SqEuclidean(), x) ./ 2)
pw(::EQ, x::AV) = exp.(.-pw(SqEuclidean(), x) ./ 2)



@doc raw"""
    PerEQ <: BaseKernel

The usual periodic kernel derived by mapping the input domain onto the unit circle.

`` k(x, x^\prime) = \exp (-2 (\sin (\pi | x - x^\prime |) / l)^2)``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct PerEQ{LT<:Parameter{<:Real}} <: BaseKernel
    l::LT
end
PerEQ(l::Real) = PerEQ(Parameter(l, Val(:pos)))

_pereq(d, l) = exp(-2.0*sin(π*d)^2 / l^2)

# Binary methods.
ew(k::PerEQ, x::AV, x′::AV) = _pereq.(ew(Euclidean(), x, x′), value(k.l))
pw(k::PerEQ, x::AV, x′::AV) = _pereq.(pw(Euclidean(), x, x′), value(k.l))

# Unary methods.
ew(k::PerEQ, x::AV) = _pereq.(ew(Euclidean(), x), value(k.l))
pw(k::PerEQ, x::AV) = _pereq.(pw(Euclidean(), x), value(k.l))


@doc raw"""
    Matern12 <: BaseKernel

The standardised Matern-1/2 / Exponential kernel:

`` k(x, x^\prime) = \exp(-||x - x^\prime||_2)``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern12 <: BaseKernel end

# Binary methods
ew(k::Matern12, x::AV, x′::AV) = exp.(.-ew(Euclidean(), x, x′))
pw(k::Matern12, x::AV, x′::AV) = exp.(.-pw(Euclidean(), x, x′))

# Unary methods
ew(::Matern12, x::AV) = exp.(.-ew(Euclidean(), x))
pw(::Matern12, x::AV) = exp.(.-pw(Euclidean(), x))



"""
    Exp <: Kernel

The standardised Exponential kernel. Equivalent to [`Matern12`](@ref).

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
const Exp = Matern12



"""
    Matern32 <: BaseKernel

The standardised Matern kernel with ν = 3 / 2.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern32 <: BaseKernel end

function _matern32(d::Real)
    d = sqrt(3) * d
    return (1 + d) * exp(-d)
end

# Binary methods
ew(k::Matern32, x::AV, x′::AV) = _matern32.(ew(Euclidean(), x, x′))
pw(k::Matern32, x::AV, x′::AV) = _matern32.(pw(Euclidean(), x, x′))

# Unary methods
ew(k::Matern32, x::AV) = _matern32.(ew(Euclidean(), x))
pw(k::Matern32, x::AV) = _matern32.(pw(Euclidean(), x))



"""
    Matern52 <: BaseKernel

The standardised Matern kernel with ν = 5 / 2.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern52 <: BaseKernel end

function _Matern52(d::Real)
    λ = sqrt(5) * d
    return (1 + λ + λ^2 / 3) * exp(-λ)
end

_Matern52(d::AbstractArray{<:Real}) = _Matern52.(d)

@adjoint function _Matern52(d::AbstractArray{<:Real})
    λ = sqrt(5) .* d
    b = exp.(-λ)
    return (1 .+ λ .+ λ.^2 ./ 3) .* b, Δ->(.-Δ .* sqrt(5) .* b .* λ .* (1 .+ λ) ./ 3,)
end

# Binary methods
ew(k::Matern52, x::AV, x′::AV) = _Matern52(ew(Euclidean(), x, x′))
pw(k::Matern52, x::AV, x′::AV) = _Matern52(pw(Euclidean(), x, x′))

# Unary methods
ew(k::Matern52, x::AV) = _Matern52(ew(Euclidean(), x))
pw(k::Matern52, x::AV) = _Matern52(pw(Euclidean(), x))



"""
    RQ <: BaseKernel

The standardised Rational Quadratic, with kurtosis `α`.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct RQ{Tα<:Parameter{<:Real}} <: BaseKernel
    α::Tα
end
RQ(α::Real) = RQ(Parameter(α, Val(:pos)))

_rq(d, α) = (1 + d / (2α))^(-α)

# I redefine adjoint function for `_rq`, since on my computer, Zygote's gradient 
# of `_rq` returns a number of Complex type.
Zygote.@adjoint function _rq(d::dT, α::αT) where {dT<:Real, αT<:Real}
    y = _rq(d, α)
    return y, function (ȳ)
    	T = promote_type(dT, αT)
    	x = 1 + d / (2α)
    	-0.5*ȳ*y/x, ȳ*y*(d / (x*(2α)) - log(x+eps(T)))
    	end
end

# Binary methods.
ew(k::RQ, x::AV, x′::AV) = _rq.(ew(SqEuclidean(), x, x′), value(k.α))
pw(k::RQ, x::AV, x′::AV) = _rq.(pw(SqEuclidean(), x, x′), value(k.α))

# Unary methods.
ew(k::RQ, x::AV) = _rq.(ew(SqEuclidean(), x), value(k.α))
pw(k::RQ, x::AV) = _rq.(pw(SqEuclidean(), x), value(k.α))

"""
    Cosine <: BaseKernel

Cosine BaseKernel with period parameter `p`.
"""
struct Cosine{Tp<:Parameter{<:Real}} <: BaseKernel
    p::Tp
end
Cosine(p::Real) = Cosine(Parameter(p, Val(:pos)))

# Binary methods.
function ew(k::Cosine, x::AV{<:Real}, x′::AV{<:Real})
    return cos.(π .* ew(Euclidean(), x, x′) ./ value(k.p))
end
function pw(k::Cosine, x::AV{<:Real}, x′::AV{<:Real})
    return cos.(π .* pw(Euclidean(), x, x′) ./ value(k.p))
end

# Unary methods.
ew(k::Cosine, x::AV{<:Real}) = 1 .+ ew(Euclidean(), x)
pw(k::Cosine, x::AV{<:Real}) = cos.(π .* pw(Euclidean(), x) ./ value(k.p))


"""
    Linear{T<:Real} <: BaseKernel

The standardised linear kernel / dot-product kernel.
"""
struct Linear <: BaseKernel end

# Binary methods
ew(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′
pw(k::Linear, x::AV{<:Real}, x′::AV{<:Real}) = x .* x′'
ew(k::Linear, x::ColVecs, x′::ColVecs) = reshape(sum(x.X .* x′.X; dims=1), :)
pw(k::Linear, x::ColVecs, x′::ColVecs) = x.X' * x′.X

# Unary methods
ew(k::Linear, x::AV{<:Real}) = x.^2
pw(k::Linear, x::AV{<:Real}) = x .* x'
ew(k::Linear, x::ColVecs) = reshape(sum(abs2.(x.X); dims=1), :)
pw(k::Linear, x::ColVecs) = x.X' * x.X



"""
    Poly{Tσ<:Real} <: BaseKernel

Inhomogeneous Polynomial kernel. `Poly(p, σ²)` creates a `Poly{p}` with variance σ²,
defined as
```julia
k(xl, xr) = (dot(xl, xr) + σ²)^p
```
"""
struct Poly{p, Tσ²<:Parameter{<:Real}} <: BaseKernel
    σ²::Tσ²
end
Poly(p::Int, σ²::T) where {T} = Poly{p, Parameter{T}}(Parameter(σ², Val(:pos)))

_poly(k, σ², p) = (σ² + k)^p
Zygote.@adjoint function _poly(k, σ², p)
    y = _poly(k, σ², p)
    return y, function(Δ)
        d = Δ * p * y / (σ² + k)
        return (d, d, nothing)
    end
end

# Binary methods
ew(k::Poly{p}, x::AV, x′::AV) where {p} = _poly.(ew(Linear(), x, x′), value(k.σ²), p)
pw(k::Poly{p}, x::AV, x′::AV) where {p} = _poly.(pw(Linear(), x, x′), value(k.σ²), p)

# Unary methods
ew(k::Poly{p}, x::AV) where {p} = _poly.(ew(Linear(), x), value(k.σ²), p)
pw(k::Poly{p}, x::AV) where {p} = _poly.(pw(Linear(), x), value(k.σ²), p)


"""
    GammaExp <: BaseKernel

The γ-Exponential kernel, 0 < γ ⩽ 2, is given by `k(xl, xr) = exp(-||xl - xr||^γ)`.
"""
struct GammaExp{Tγ<:Parameter{<:Real}} <: BaseKernel
    γ::Tγ
end
GammaExp(γ::T) where {T} = GammaExp(Parameter(γ, inv(Logit(zero(T), T(2.0)))))

# Binary methods
ew(k::GammaExp, x::AV, x′::AV) = exp.(.-ew(Euclidean(), x, x′).^(value(k.γ)))
pw(k::GammaExp, x::AV, x′::AV) = exp.(.-pw(Euclidean(), x, x′).^(value(k.γ)))

# Unary methods
ew(k::GammaExp, x::AV) = exp.(.-ew(Euclidean(), x).^(value(k.γ)))
pw(k::GammaExp, x::AV) = exp.(.-pw(Euclidean(), x).^(value(k.γ)))


"""
    Wiener <: BaseKernel

The standardised stationary Wiener-process kernel.
"""
struct Wiener <: BaseKernel end

_wiener(x::Real, x′::Real) = min(x, x′)

# Binary methods
ew(k::Wiener, x::AV{<:Real}, x′::AV{<:Real}) = _wiener.(x, x′)
pw(k::Wiener, x::AV{<:Real}, x′::AV{<:Real}) = _wiener.(x, x′')

# Unary methods
ew(k::Wiener, x::AV{<:Real}) = x
pw(k::Wiener, x::AV{<:Real}) = pw(k, x, x)



"""
    WienerVelocity <: BaseKernel

The standardised WienerVelocity kernel.
"""
struct WienerVelocity <: BaseKernel end

_wiener_vel(x::Real, x′::Real) = min(x, x′)^3 / 3 + abs(x - x′) * min(x, x′)^2 / 2

# Binary methods
ew(k::WienerVelocity, x::AV{<:Real}, x′::AV{<:Real}) = _wiener_vel.(x, x′)
pw(k::WienerVelocity, x::AV{<:Real}, x′::AV{<:Real}) = _wiener_vel.(x, x′')

# Unary methods
ew(k::WienerVelocity, x::AV{<:Real}) = ew(k, x, x)
pw(k::WienerVelocity, x::AV{<:Real}) = pw(k, x, x)



"""
    Noise{T<:Real} <: BaseKernel

The standardised aleatoric white-noise kernel. Isn't really a kernel, but never mind...
"""
struct Noise{T<:Real} <: BaseKernel end
Noise() = Noise{Int}()

# Binary methods.
ew(k::Noise{T}, x::AV, x′::AV) where {T} = zeros(T, broadcast_shape(size(x), size(x′))...)
pw(k::Noise{T}, x::AV, x′::AV) where {T} = zeros(T, length(x), length(x′))

# Unary methods.
ew(k::Noise{T}, x::AV) where {T} = ones(T, length(x))
pw(k::Noise{T}, x::AV) where {T} = diagm(0=>ones(T, length(x)))


"""
    Precomputed{T<:Real} <: Kernel

Using the values of a precomputed Gram matrix as a kernel.

Optionally checks if the Gram matrix is positive definite by setting
`checkpd=true`
"""
struct Precomputed{M<:AbstractMatrix{<:Real}} <: Kernel
    K::M
    function Precomputed(K::AbstractMatrix{<:Real}; checkpd=false)
        checksquare(K)
        checkpd && @assert isposdef(K) "M is not positive definite"
        return new{typeof(K)}(K)
    end
end

# Binary methods.
ew(k::Precomputed, x::AV{<:Integer}, x′::AV{<:Integer}) = [k.K[p, q] for (p, q) in zip(x, x′)]
pw(k::Precomputed, x::AV{<:Integer}, x′::AV{<:Integer}) = k.K[x, x′]

# Unary methods.
ew(k::Precomputed, x::AV{<:Integer}) = diag(k.K)[x]
pw(k::Precomputed, x::AV{<:Integer}) = k.K[x,x]

precomputed(K::AbstractMatrix) = Precomputed(K)
export precomputed



#
# Composite Kernels
#

"""
    Sum{Tkl<:Kernel, Tkr<:Kernel} <: Kernel

Represents the sum of two kernels `kl` and `kr` s.t. `k(x, x′) = kl(x, x′) + kr(x, x′)`.
"""
struct Sum{Tkl<:Kernel, Tkr<:Kernel} <: Kernel
    kl::Tkl
    kr::Tkr
end
child(s::Sum) = (s.kl, s.kr)
"""
    +(kl::Kernel, kr::Kernel)

Construct the kernel whose value is given by the sum of those of `kl` and `kr`.

```jldoctest
julia> kl, kr = EQ(), Matern32();

julia> x = randn(11);

julia> pw(kl + kr, x) == pw(kl, x) + pw(kr, x)
true
```
"""
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
child(p::Product) = (p.kl, p.kr)
"""
    *(kl::Kernel, kr::Kernel)

Construct the kernel whose value is given by the product of those of `kl` and `kr`.

```jldoctest
julia> kl, kr = EQ(), Matern32();

julia> x = randn(11);

julia> pw(kl * kr, x) == pw(kl, x) .* pw(kr, x)
true
```
"""
*(kl::Kernel, kr::Kernel) = Product(kl, kr)

# Binary methods
ew(k::Product, x::AV, x′::AV) = ew(k.kl, x, x′) .* ew(k.kr, x, x′)
pw(k::Product, x::AV, x′::AV) = pw(k.kl, x, x′) .* pw(k.kr, x, x′)

# Unary methods
ew(k::Product, x::AV) = ew(k.kl, x) .* ew(k.kr, x)
pw(k::Product, x::AV) = pw(k.kl, x) .* pw(k.kr, x)



"""
Scaled{Tσ²<:AV{<:Real}, Tk<:Kernel} <: Kernel

Scale the variance of `Kernel` `k` by `σ²` s.t. `(σ² * k)(x, x′) = σ² * k(x, x′)`.
"""
struct Scaled{Tσ²<:Parameter{<:Real}, Tk<:Kernel} <: Kernel
    σ²::Tσ²
    k::Tk
end
scale(k::Kernel, σ²::Real) = Scaled(Parameter(σ², Val(:pos)), k)

"""
    *(σ²::Real, k::Kernel)
    *(k::Kernel, σ²::Real)

The right way to choose the variance of a kernel. Specifically, construct a kernel that
scales the output of `k` by `σ²`:
```jldoctest
julia> k = EQ();

julia> x = randn(11);

julia> pw(0.5 * k, x) == 0.5 .* Stheno.pw(k, x)
true
```
"""
*(σ²::Real, k::Kernel) = scale(k, σ²)
*(k::Kernel, σ²) = σ² * k

# Binary methods.
ew(k::Scaled, x::AV, x′::AV) = value(k.σ²) .* ew(k.k, x, x′)
pw(k::Scaled, x::AV, x′::AV) = value(k.σ²) .* pw(k.k, x, x′)

# Unary methods.
ew(k::Scaled, x::AV) = value(k.σ²) .* ew(k.k, x)
pw(k::Scaled, x::AV) = value(k.σ²) .* pw(k.k, x)


"""
    Stretched{Tk<:Kernel} <: Kernel

Apply a length scale to a kernel. Specifically, `k(x, x′) = k(x/a, x′/a)`.
"""
struct Stretched{Ta<:Parameter{<:Real}, Tk<:Kernel} <: Kernel
    a::Ta
    k::Tk
end
"""
    stretch(k::Kernel, a::Union{Real, AbstractVecOrMat{<:Real})

The canonical way to give a notion of length-scale to a kernel, `a::Real`, to construct
an automatic relevance determination (ARD) kernel, `a::AbstractVector{<:Real}`, or a
discriminative factor-analysis kernel, `a::AbstractMatrix{<:Real}`.

# Length Scale

For `a::Real`, return the kernel given by:
```julia
stretch(k, a)(x, y) = k(a .* x, a .* y)
```
where `x` and `y` are both either `Real`s or `AbstractVector{<:Real}`s. e.g.
```jldoctest
xs = range(0.0, 10.0; length=2)
ys = range(0.5, 10.5; length=3)
k = stretch(EQ(), 0.5)
K = pairwise(k, xs, ys)

# output
2×3 Array{Float64,2}:
 0.969233    0.0227942  1.03485e-6
 1.26071e-5  0.0795595  0.969233
```
if `x` and `y` are `Real`s.

# Automatic Relevance Determination (ARD)

For `a::AbstractVector{<:Real}`, return the automatic relevance determination (ARD) kernel
given by:
```julia
stretch(k, a)(x, y) = k(a .* x, a .* y)
```
where `x` and `y` must be `AbstractVector{<:Real}`s. e.g.

```jldoctest
rng = MersenneTwister(123456)
xs = ColVecs(randn(rng, 2, 2)) # efficient vector-of-column-vectors
ys = ColVecs(randn(rng, 2, 3)) # efficient vector-of-column-vectors
length_scales = [0.1 0.5]
k = stretch(EQ(), 1 ./ length_scales)
K = pairwise(k, xs, ys)

# output
2×3 Array{Float64,2}:
 1.00858e-8   0.000170346  0.0439152
 2.79756e-15  2.0696e-74   9.91334e-31
```

# Discriminative Factor Analysis

For `a::AbstractMatrix{<:Real}`, return the kernel given by:
```julia
stretch(k, a)(x, y) = k(a * x, a * y)
```
where  `x` and `y` must be `AbstractVector{<:Real}`s. e.g.

```jldoctest
rng = MersenneTwister(123456)

# Efficiently represented vectors of column-vectors
xs = ColVecs(randn(rng, 5, 2)) 
ys = ColVecs(randn(rng, 5, 3))

# Project from 5 dimensions down onto 2
A = rand(rng, 2, 5)

# Construct kernel and compute covariance matrix between `xs` and `ys`.
k = stretch(EQ(), A)
K = pairwise(k, xs, ys)

# output
2×3 Array{Float64,2}:
 0.000289928  0.00519575  0.000127567 
 0.0507073    0.495861    0.432123
```
"""
stretch(k::Kernel, a::Union{Real, AVM{<:Real}}) = Stretched(Parameter(a, Val(:pos)), k)

# Binary methods (scalar `a`, scalar-valued input)
function ew(k::Stretched{<:Parameter{<:Real, 0}}, x::AV{<:Real}, x′::AV{<:Real})
    return ew(k.k, value(k.a) .* x, value(k.a) .* x′)
end
function pw(k::Stretched{<:Parameter{<:Real, 0}}, x::AV{<:Real}, x′::AV{<:Real})
    return pw(k.k, value(k.a) .* x, value(k.a) .* x′)
end

# Unary methods (scalar `a`, scalar-valued input)
ew(k::Stretched{<:Parameter{<:Real, 0}}, x::AV{<:Real}) = ew(k.k, value(k.a) .* x)
pw(k::Stretched{<:Parameter{<:Real, 0}}, x::AV{<:Real}) = pw(k.k, value(k.a) .* x)

# Binary methods (scalar/vector `a`, vector-valued input)
function ew(k::Stretched{<:Union{Parameter{<:Real, 0}, Parameter{<:Real, 1}}}, x::ColVecs, x′::ColVecs)
    return ew(k.k, ColVecs(value(k.a) .* x.X), ColVecs(value(k.a) .* x′.X))
end
function pw(k::Stretched{<:Union{Parameter{<:Real, 0}, Parameter{<:Real, 1}}}, x::ColVecs, x′::ColVecs)
    return pw(k.k, ColVecs(value(k.a) .* x.X), ColVecs(value(k.a) .* x′.X))
end

# Unary methods (vector `a`, vector-valued input)
ew(k::Stretched{<:Union{Parameter{<:Real, 0}, Parameter{<:Real, 1}}}, x::ColVecs) = ew(k.k, ColVecs(value(k.a) .* x.X))
pw(k::Stretched{<:Union{Parameter{<:Real, 0}, Parameter{<:Real, 1}}}, x::ColVecs) = pw(k.k, ColVecs(value(k.a) .* x.X))

# Binary methods (matrix `a`, vector-valued input)
function ew(k::Stretched{<:Parameter{<:Real, 2}}, x::ColVecs, x′::ColVecs)
    return ew(k.k, ColVecs(value(k.a) * x.X), ColVecs(value(k.a) * x′.X))
end
function pw(k::Stretched{<:Parameter{<:Real, 2}}, x::ColVecs, x′::ColVecs)
    return pw(k.k, ColVecs(value(k.a) * x.X), ColVecs(value(k.a) * x′.X))
end

# Unary methods (matrix `a`, vector-valued input)
ew(k::Stretched{<:Parameter{<:Real, 2}}, x::ColVecs) = ew(k.k, ColVecs(value(k.a) * x.X))
pw(k::Stretched{<:Parameter{<:Real, 2}}, x::ColVecs) = pw(k.k, ColVecs(value(k.a) * x.X))


"""
    kernel(k::Kernel;  l::Real=nothing, s::Real=nothing)

Convenience functionality to provide a kernel with a length scale `l`, and to scale the
variance of `k` by `s`. Simply applies the [`stretch`](@ref stretch) and [`*`](@ref *)
functions.

```jldoctest
julia> k1 = kernel(EQ(); l=1.1, s=0.9);

julia> k2 = 0.9 * stretch(EQ(), 1 / 1.1);

julia> x = randn(11);

julia> pw(k1, x) == pw(k2, x)
true
```
"""
function kernel(k::Kernel; l::Union{Real, Nothing}=nothing, s::Union{Real, Nothing}=nothing)
    if l !== nothing
        k = stretch(k, 1 / l)
    end
    if s !== nothing
        k = s * k
    end
    return k
end
