export GPC

abstract type AbstractGP end

# A collection of GPs (GPC == "GP Collection"). Used to keep track of GPs.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

@nograd GPC

next_index(gpc::GPC) = gpc.n + 1

#
# Projecting AbstractGPs onto a finite dimensional marginal
#

import Base: rand, length
import Distributions: logpdf, ContinuousMultivariateDistribution

export mean, std, cov, marginals, rand, logpdf, elbo, dtc

"""
    FiniteGP{Tf<:AbstractGP, Tx<:AV, TΣy}

The finite-dimensional projection of the AbstractGP `f` at `x`.
"""
struct FiniteGP{Tf<:AbstractGP, Tx<:AV, TΣy} <: ContinuousMultivariateDistribution
    f::Tf
    x::Tx 
    Σy::TΣy
end
FiniteGP(f::AbstractGP, x::AV, σ²::AV{<:Real}) = FiniteGP(f, x, Diagonal(σ²))
FiniteGP(f::AbstractGP, x::AV, σ²::Real) = FiniteGP(f, x, Fill(σ², length(x)))
FiniteGP(f::AbstractGP, x::AV) = FiniteGP(f, x, 1e-18)

length(f::FiniteGP) = length(f.x)

"""
    mean(fx::FiniteGP)

Compute the mean vector of `fx`.

```jldoctest
julia> f = GP(Matern52(), GPC());

julia> x = randn(11);

julia> mean(f(x)) == zeros(11)
true
```
"""
mean(fx::FiniteGP) = mean_vector(fx.f, fx.x)

"""
    cov(f::FiniteGP)

Compute the covariance matrix of `fx`.

## Noise-free observations

```jldoctest cov_finitegp
julia> f = GP(Matern52(), GPC());

julia> x = randn(11);

julia> # Noise-free

julia> cov(f(x)) == Stheno.pw(Matern52(), x)
true
```

## Isotropic observation noise

```jldoctest cov_finitegp
julia> cov(f(x, 0.1)) == Stheno.pw(Matern52(), x) + 0.1 * I
true
```

## Independent anisotropic observation noise

```jldoctest cov_finitegp
julia> s = rand(11);

julia> cov(f(x, s)) == Stheno.pw(Matern52(), x) + Diagonal(s)
true
```

## Correlated observation noise

```jldoctest cov_finitegp
julia> A = randn(11, 11); S = A'A;

julia> cov(f(x, S)) == Stheno.pw(Matern52(), x) + S
true
```
"""
cov(f::FiniteGP) = cov(f.f, f.x) + f.Σy

"""
    cov(fx::FiniteGP, gx::FiniteGP)

Compute the cross-covariance matrix between `fx` and `gx`.

```jldoctest
julia> f = GP(Matern32(), GPC());

julia> x1 = randn(11);

julia> x2 = randn(13);

julia> cov(f(x1), f(x2)) == pw(Matern32(), x1, x2)
true
```
"""
cov(fx::FiniteGP, gx::FiniteGP) = cov(fx.f, gx.f, fx.x, gx.x)

"""
    marginals(f::FiniteGP)

Compute a vector of Normal distributions representing the marginals of `f` efficiently.
In particular, the off-diagonal elements of `cov(f(x))` are never computed.

```jldoctest
julia> f = GP(Matern32(), GPC());

julia> x = randn(11);

julia> fs = marginals(f(x));

julia> mean.(fs) == mean(f(x))
true

julia> std.(fs) == sqrt.(diag(cov(f(x))))
true
```
"""
marginals(f::FiniteGP) = Normal.(mean(f), sqrt.(cov_diag(f.f, f.x) .+ diag(f.Σy)))

"""
    rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

Obtain `N` independent samples from the marginals `f` using `rng`. Single-sample methods
produce a `length(f)` vector. Multi-sample methods produce a `length(f)` x `N` `Matrix`.

```jldoctest
julia> f = GP(Matern32(), GPC());

julia> x = randn(11);

julia> rand(f(x)) isa Vector{Float64}
true

julia> rand(MersenneTwister(123456), f(x)) isa Vector{Float64}
true

julia> rand(f(x), 3) isa Matrix{Float64}
true

julia> rand(MersenneTwister(123456), f(x), 3) isa Matrix{Float64}
true
```
"""
function rand(rng::AbstractRNG, f::FiniteGP, N::Int)
    μ, C = mean(f), cholesky(Symmetric(cov(f)))
    return μ .+ C.U' * randn(rng, promote_type(eltype(μ), eltype(C)), length(μ), N)
end
rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(rng::AbstractRNG, f::FiniteGP) = vec(rand(rng, f, 1))
rand(f::FiniteGP) = vec(rand(f, 1))

"""
    logpdf(f::FiniteGP, y::AbstractVecOrMat{<:Real})

The logpdf of `y` under `f` if is `y isa AbstractVector`. logpdf of each column of `y` if
`y isa Matrix`.

```jldoctest
julia> f = GP(Matern32(), GPC());

julia> x = randn(11);

julia> y = rand(f(x));

julia> logpdf(f(x), y) isa Real
true

julia> Y = rand(f(x), 3);

julia> logpdf(f(x), Y) isa AbstractVector{<:Real}
true
```
"""
logpdf(f::FiniteGP, y::AbstractVector{<:Real}) = first(logpdf(f, reshape(y, :, 1)))

function logpdf(f::FiniteGP, Y::AbstractMatrix{<:Real})
    μ, C = mean(f), cholesky(Symmetric(cov(f)))
    T = promote_type(eltype(μ), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ diag_Xt_invA_X(C, Y .- μ)) ./ 2
end

"""
   elbo(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)

The saturated Titsias Evidence LOwer Bound (ELBO) [1]. `y` are observations of `f`, and `u`
are pseudo-points.

```jldoctest
julia> f = GP(Matern52(), GPC());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=13);

julia> y = rand(f(x, 0.1));

julia> elbo(f(x, 0.1), y, f(z)) < logpdf(f(x, 0.1), y)
true
```

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    _dtc, chol_Σy, A = _compute_intermediates(f, y, u)
    return _dtc - (tr_Cf_invΣy(f, f.Σy, chol_Σy) - sum(abs2, A)) / 2
end

"""
    dtc(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)

The Deterministic Training Conditional (DTC) [1]. `y` are observations of `f`, and `u`
are pseudo-points.

```jldoctest
julia> f = GP(Matern52(), GPC());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=256);

julia> y = rand(f(x, 0.1));

julia> isapprox(dtc(f(x, 0.1), y, f(z)), logpdf(f(x, 0.1), y); atol=1e-3, rtol=1e-3)
true
```

[1] - M. Seeger, C. K. I. Williams and N. D. Lawrence. "Fast Forward Selection to Speed Up
Sparse Gaussian Process Regression". In: Proceedings of the Ninth International Workshop on
Artificial Intelligence and Statistics. 2003
"""
dtc(f::FiniteGP, y::AV{<:Real}, u::FiniteGP) = first(_compute_intermediates(f, y, u))

# Factor out computations common to the `elbo` and `dtc`.
function _compute_intermediates(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    consistency_check(f, y, u)
    chol_Σy = cholesky(f.Σy)

    A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
    Λ_ε = cholesky(Symmetric(A * A' + I))
    δ = chol_Σy.U' \ (y - mean(f))

    tmp = logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ))
    _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
    return _dtc, chol_Σy, A
end


function consistency_check(f, y, u)
    @assert length(f) == size(y, 1)
end
Zygote.@nograd consistency_check

import Base: \ 
\(A::AbstractMatrix, B::Diagonal) = A \ Matrix(B)

\(A::Union{LowerTriangular, UpperTriangular}, B::Diagonal) = A \ Matrix(B)
\(A::Adjoint{<:Any, <:Union{LowerTriangular, UpperTriangular}}, B::Diagonal) = A \ Matrix(B)


# Compute tr(Cf / Σy) efficiently for different types of Σy. For dense Σy you obviously need
# to compute the entirety of Cf, which is bad, but for particular structured Σy one requires
# only a subset of the elements. Σy isa UniformScaling is version usually considered.
function tr_Cf_invΣy(f::FiniteGP, Σy::UniformScaling, chol_Σy::Cholesky)
    return sum(cov_diag(f.f, f.x)) / Σy.λ
end
function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal, chol_Σy::Cholesky)
    return sum(cov_diag(f.f, f.x) ./ diag(Σy))
end
tr_Cf_invΣy(f::FiniteGP, Σy::Matrix, chol_Σy::Cholesky) = tr(chol_Σy \ cov(f))

# function tr_Cf_invΣy(f::FiniteGP, Σy::BlockDiagonal, chol_Σy::Cholesky)
#     C = cholesky(Symmetric(_get_kernel_block_diag(f, cumulsizes(Σy, 1))))
#     return tr_At_A(chol_Σy.U' \ C.U')
# end

# function _get_kernel_block_diag(f::FiniteGP, cs)
#     k = kernel(f.f)
#     ids = map(n->cs[n]:cs[n+1]-1, 1:length(cs)-1)
#     xs = map(id->f.x[id], ids)
#     Σs = map(x->pw(k, x), xs)
#     return block_diagonal(Σs)
# end

# function _get_kernel_block_diag(f::FiniteGP{<:GP{<:BlockMean, <:BlockKernel}, <:BlockData}, cs)
#     k = kernel(f.f)
#     ids = map(n->cs[n]:cs[n+1]-1, 1:length(cs)-1)
#     @assert _test_block_consistency(ids, f)
#     xs = blocks(f.x)
#     Σs = map(n->pw(k.ks[n], xs[n]), 1:length(xs))
#     return block_diagonal(Σs)
# end

_test_block_consistency(ids, f) = length.(ids) == length.(blocks(f.x))
Zygote.@nograd _test_block_consistency

# """
#     elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP, mε::AV{<:Real}, Λε::AM{<:Real})

# The unsaturated Titsias-ELBO.
# """
# function elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP, mε::AV{<:Real}, Λε::AM{<:Real})
#     @assert length(u.x) == length(mε)
#     @assert size(Λε) == (length(mε), length(mε))
#     # do stuff.
# end

import Base: |, merge
export ←, |, Obs

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation{Tf<:FiniteGP, Ty<:Vector}
    f::Tf
    y::Ty
end

const Obs = Observation


←(f, y) = Observation(f, y)
get_f(c::Observation) = c.f
get_y(c::Observation) = c.y
