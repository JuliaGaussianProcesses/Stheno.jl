import Base: +, *, zero, cos
using Distances: sqeuclidean, SqEuclidean, Euclidean
using Base.Broadcast: broadcast_shape
using LinearAlgebra: isposdef, checksquare

abstract type Kernel end

# API exports
export Kernel, kernel, elementwise, pairwise, ew, pw, stretch, scale

# Kernel exports
export EQ, Exp, PerEQ, Matern12, Matern32, Matern52, RQ, Cosine, Linear, Poly, GammaExp, Wiener,
    WienerVelocity, Precomputed



#
# Base Kernels
#

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



@doc raw"""
    EQ() <: Kernel

The standardised Exponentiated Quadratic kernel. a.k.a. the Radial Basis Function (RBF), or
Squared Exponential kernel.

``k(x, x^\prime) = \exp( -\frac{1}{2} || x - x^\prime||_2^2 )``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct EQ <: Kernel end

# Binary methods.
ew(::EQ, x::AV, x′::AV) = exp.(.-ew(SqEuclidean(), x, x′) ./ 2)
pw(::EQ, x::AV, x′::AV) = exp.(.-pw(SqEuclidean(), x, x′) ./ 2)

# Unary methods.
ew(::EQ, x::AV) = exp.(.-ew(SqEuclidean(), x) ./ 2)
pw(::EQ, x::AV) = exp.(.-pw(SqEuclidean(), x) ./ 2)



@doc raw"""
    PerEQ

The usual periodic kernel derived by mapping the input domain onto the unit circle.

`` k(x, x^\prime) = \exp (-2 \sin (\pi | x - x^\prime |^2)``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct PerEQ{LT<:AV{<:Real}, fT} <: Kernel
    l::LT
    f::fT
end
PerEQ(l::Real, f) = PerEQ(typeof(l)[l], f)
PerEQ(l::Real) = PerEQ(l, identity)

_pereq(d, l) = exp(-2.0*sin(π*d)^2 / l^2)

# Binary methods.
ew(k::PerEQ, x::AV, x′::AV) = _pereq.(ew(Euclidean(), x, x′), k.f(k.l[1]))
pw(k::PerEQ, x::AV, x′::AV) = _pereq.(pw(Euclidean(), x, x′), k.f(k.l[1]))

# Unary methods.
ew(k::PerEQ, x::AV) = _pereq.(ew(Euclidean(), x), k.f(k.l[1]))
pw(k::PerEQ, x::AV) = _pereq.(pw(Euclidean(), x), k.f(k.l[1]))


@doc raw"""
    Matern12 <: Kernel

The standardised Matern-1/2 / Exponential kernel:

`` k(x, x^\prime) = \exp(-||x - x^\prime||_2)``

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern12 <: Kernel end

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
    Matern32 <: Kernel

The standardised Matern kernel with ν = 3 / 2.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern32 <: Kernel end

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
    Matern52 <: Kernel

The standardised Matern kernel with ν = 5 / 2.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct Matern52 <: Kernel end

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
    RQ <: Kernel

The standardised Rational Quadratic, with kurtosis `α`.

For length scales etc see [`stretch`](@ref), for variance see [`*`](@ref).
"""
struct RQ{Tα<:AV{<:Real}, fT} <: Kernel
    α::Tα
    f::fT
end
RQ(α::Real, f) = RQ(typeof(α)[α], f)
RQ(α::Real) = RQ(α, identity)

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
ew(k::RQ, x::AV, x′::AV) = _rq.(ew(SqEuclidean(), x, x′), k.f(k.α[1]))
pw(k::RQ, x::AV, x′::AV) = _rq.(pw(SqEuclidean(), x, x′), k.f(k.α[1]))

# Unary methods.
ew(k::RQ, x::AV) = _rq.(ew(SqEuclidean(), x), k.f(k.α[1]))
pw(k::RQ, x::AV) = _rq.(pw(SqEuclidean(), x), k.f(k.α[1]))


"""
    Cosine <: Kernel

Cosine Kernel with period parameter `p`.
"""
struct Cosine{Tp<:AV{<:Real}, fT} <: Kernel
    p::Tp
    f::fT
end
Cosine(p::Real, f) = Cosine(typeof(p)[p], f)
Cosine(p::Real) = Cosine(p, identity)

# Binary methods.
ew(k::Cosine, x::AV{<:Real}, x′::AV{<:Real}) = cos.(pi.*ew(Euclidean(), x, x′) ./ k.f(k.p[1]))
pw(k::Cosine, x::AV{<:Real}, x′::AV{<:Real}) = cos.(pi.*pw(Euclidean(), x, x′) ./ k.f(k.p[1]))

# Unary methods.
ew(k::Cosine, x::AV{<:Real}) = 1 .+ ew(Euclidean(), x)
pw(k::Cosine, x::AV{<:Real}) = cos.(pi .* pw(Euclidean(), x) ./ k.f(k.p[1]))


"""
    Linear{T<:Real} <: Kernel

The standardised linear kernel / dot-product kernel.
"""
struct Linear <: Kernel end

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
    Poly{Tσ<:Real} <: Kernel

Inhomogeneous Polynomial kernel. `Poly(p, σ²)` creates a `Poly{p}` with variance σ²,
defined as
```julia
k(xl, xr) = (dot(xl, xr) + σ²)^p
```
"""
struct Poly{p, Tσ²<:AV{<:Real}, fT} <: Kernel
    σ²::Tσ²
    f::fT
end
Poly(p::Int, σ²::Real, f) = Poly{p, AV{typeof(σ²)}, typeof(f)}(typeof(σ²)[σ²], f)
Poly(p::Int, σ²::Real) = Poly(p, σ², identity)

_poly(k, σ², p) = (σ² + k)^p
Zygote.@adjoint function _poly(k, σ², p)
    y = _poly(k, σ², p)
    return y, function(Δ)
        d = Δ * p * y / (σ² + k)
        return (d, d, nothing)
    end
end

# Binary methods
ew(k::Poly{p}, x::AV, x′::AV) where {p} = _poly.(ew(Linear(), x, x′), k.f(k.σ²[1]), p)
pw(k::Poly{p}, x::AV, x′::AV) where {p} = _poly.(pw(Linear(), x, x′), k.f(k.σ²[1]), p)

# Unary methods
ew(k::Poly{p}, x::AV) where {p} = _poly.(ew(Linear(), x), k.f(k.σ²[1]), p)
pw(k::Poly{p}, x::AV) where {p} = _poly.(pw(Linear(), x), k.f(k.σ²[1]), p)


"""
    GammaExp

The γ-Exponential kernel, 0 < γ ⩽ 2, is given by `k(xl, xr) = exp(-||xl - xr||^γ)`.
"""
struct GammaExp{Tγ<:AV{<:Real}, fT} <: Kernel
    γ::Tγ
    f::fT
end
GammaExp(γ::Real, f) = GammaExp(typeof(γ)[γ], f)
GammaExp(γ::Real) = GammaExp(γ, identity)

# Binary methods
ew(k::GammaExp, x::AV, x′::AV) = exp.(.-ew(Euclidean(), x, x′).^(k.f(k.γ[1])))
pw(k::GammaExp, x::AV, x′::AV) = exp.(.-pw(Euclidean(), x, x′).^(k.f(k.γ[1])))

# Unary methods
ew(k::GammaExp, x::AV) = exp.(.-ew(Euclidean(), x).^(k.f(k.γ[1])))
pw(k::GammaExp, x::AV) = exp.(.-pw(Euclidean(), x).^(k.f(k.γ[1])))


"""
    Wiener <: Kernel

The standardised stationary Wiener-process kernel.
"""
struct Wiener <: Kernel end

_wiener(x::Real, x′::Real) = min(x, x′)

# Binary methods
ew(k::Wiener, x::AV{<:Real}, x′::AV{<:Real}) = _wiener.(x, x′)
pw(k::Wiener, x::AV{<:Real}, x′::AV{<:Real}) = _wiener.(x, x′')

# Unary methods
ew(k::Wiener, x::AV{<:Real}) = x
pw(k::Wiener, x::AV{<:Real}) = pw(k, x, x)



"""
    WienerVelocity <: Kernel

The standardised WienerVelocity kernel.
"""
struct WienerVelocity <: Kernel end

_wiener_vel(x::Real, x′::Real) = min(x, x′)^3 / 3 + abs(x - x′) * min(x, x′)^2 / 2

# Binary methods
ew(k::WienerVelocity, x::AV{<:Real}, x′::AV{<:Real}) = _wiener_vel.(x, x′)
pw(k::WienerVelocity, x::AV{<:Real}, x′::AV{<:Real}) = _wiener_vel.(x, x′')

# Unary methods
ew(k::WienerVelocity, x::AV{<:Real}) = ew(k, x, x)
pw(k::WienerVelocity, x::AV{<:Real}) = pw(k, x, x)



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
        new{typeof(K)}(K)
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

"""
    +(kl::Kernel, kr::Kernel)

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
    Scaled{Tσ²<:Real, Tk<:Kernel} <: Kernel

Scale the variance of `Kernel` `k` by `σ²` s.t. `(σ² * k)(x, x′) = σ² * k(x, x′)`.
"""
struct Scaled{Tσ²<:AV{<:Real}, Tk<:Kernel, fT} <: Kernel
    σ²::Tσ²
    k::Tk
    f::fT
end
scale(k::Kernel, σ²::Real, f) = Scaled(typeof(σ²)[σ²], k, f)
scale(k::Kernel, σ²::Real) = scale(k, σ², identity)
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
ew(k::Scaled, x::AV, x′::AV) = k.f(k.σ²[1]) .* ew(k.k, x, x′)
pw(k::Scaled, x::AV, x′::AV) = k.f(k.σ²[1]) .* pw(k.k, x, x′)

# Unary methods.
ew(k::Scaled, x::AV) = k.f(k.σ²[1]) .* ew(k.k, x)
pw(k::Scaled, x::AV) = k.f(k.σ²[1]) .* pw(k.k, x)


"""
    Stretched{Tk<:Kernel} <: Kernel

Apply a length scale to a kernel. Specifically, `k(x, x′) = k(a * x, a * x′)`.
"""
struct Stretched{Ta<:Union{AV{<:Real}, AM{<:Real}}, Tk<:Kernel, fT} <: Kernel
    a::Ta
    k::Tk
    f::fT
end
"""
    stretch(k::Kernel, a::Union{Real, AbstractVecOrMat{<:Real})

The canonical way to give a notion of length-scale to a kernel, `a::Real`, to construct
an automatic relevance determination (ARD) kernel, `a::AbstractVector{<:Real}`, or a
discriminative factor-analysis kernel, `a::AbstractMatrix{<:Real}`.

# Length Scale

For `a::Real`, return the kernel given by:
```julia
stretch(k, a)(x, y) = k(a * x, a * y)
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
A = randn(rng, 2, 5)

# Construct kernel and compute covariance matrix between `xs` and `ys`.
k = stretch(EQ(), A)
K = pairwise(k, xs, ys)

# output
2×3 Array{Float64,2}:
 0.000324131  3.77237e-12  8.12649e-16
 1.40202e-8   0.293658     0.0808585
```
"""
stretch(k::Kernel, a::AVM{<:Real}, f) = Stretched(a, k, f)
stretch(k::Kernel, a::AVM{<:Real}) = stretch(k, a, identity)
stretch(k::Kernel, a::Real, f) = stretch(k, typeof(a)[a], f)
stretch(k::Kernel, a::Real) = stretch(k, a, identity)

# Binary methods (vector `a`, scalar-valued input)
ew(k::Stretched{<:AV{<:Real}}, x::AV{<:Real}, x′::AV{<:Real}) = ew(k.k, k.f.(k.a) .* x, k.f.(k.a) .* x′)
pw(k::Stretched{<:AV{<:Real}}, x::AV{<:Real}, x′::AV{<:Real}) = pw(k.k, k.f.(k.a) .* x, k.f.(k.a) .* x′)

# Unary methods (vector `a`, scalar-valued input)
ew(k::Stretched{<:AV{<:Real}}, x::AV{<:Real}) = ew(k.k, k.f.(k.a) .* x)
pw(k::Stretched{<:AV{<:Real}}, x::AV{<:Real}) = pw(k.k, k.f.(k.a) .* x)

# Binary methods (vector `a`, vector-valued input)
function ew(k::Stretched{<:AV{<:Real}}, x::ColVecs, x′::ColVecs)
    return ew(k.k, ColVecs(k.f.(k.a) .* x.X), ColVecs(k.f.(k.a) .* x′.X))
end
function pw(k::Stretched{<:AV{<:Real}}, x::ColVecs, x′::ColVecs)
    return pw(k.k, ColVecs(k.f.(k.a) .* x.X), ColVecs(k.f.(k.a) .* x′.X))
end

# Unary methods (vector `a`, vector-valued input)
ew(k::Stretched{<:AV{<:Real}}, x::ColVecs) = ew(k.k, ColVecs(k.f.(k.a) .* x.X))
pw(k::Stretched{<:AV{<:Real}}, x::ColVecs) = pw(k.k, ColVecs(k.f.(k.a) .* x.X))

# Binary methods (matrix `a`, vector-valued input)
function ew(k::Stretched{<:AM{<:Real}}, x::ColVecs, x′::ColVecs)
    return ew(k.k, ColVecs(k.f.(k.a) * x.X), ColVecs(k.f.(k.a) * x′.X))
end
function pw(k::Stretched{<:AM{<:Real}}, x::ColVecs, x′::ColVecs)
    return pw(k.k, ColVecs(k.f.(k.a) * x.X), ColVecs(k.f.(k.a) * x′.X))
end

# Unary methods (scalar and vector `a`, vector-valued input)
ew(k::Stretched{<:AM{<:Real}}, x::ColVecs) = ew(k.k, ColVecs(k.f.(k.a) * x.X))
pw(k::Stretched{<:AM{<:Real}}, x::ColVecs) = pw(k.k, ColVecs(k.f.(k.a) * x.X))



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
