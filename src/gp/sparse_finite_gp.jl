"""
    SparseFiniteGP{T1<:FiniteGP, T2<:FiniteGP}

A finite-dimensional projection of an `AbstractGP` `f` at locations `x`, which uses a second
`FiniteGP` defined at a sparse set of inducing points [1] to do approximate inference.

This object has similar methods to an ordinary `FiniteGP`, but when you call `logpdf` on it,
it actually computes the `elbo`, and when you call `posterior` on it, you get an approximate
posterior.

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.

```jldoctest
julia> f = atomic(GP(Matern32Kernel()), GPC());

julia> fobs = f(rand(100));

julia> finducing = f(0:0.1:1);

julia> fxu = SparseFiniteGP(fobs, finducing);

julia> y = rand(fxu);

julia> logpdf(fxu, y) < logpdf(fobs, y)
true
```
"""
struct SparseFiniteGP{T1<:FiniteGP, T2<:FiniteGP} <: AbstractGPs.AbstractMvNormal
    fobs::T1
    finducing::T2
end

Base.length(f::SparseFiniteGP) = length(f.fobs)

AbstractGPs.mean(f::SparseFiniteGP) = mean(f.fobs)

const __covariance_error = "The covariance matrix of a sparse GP can often be dense and " *
    "can cause the computer to run out of memory. If you are sure you have enough " *
    "memory, you can use `cov(f.fobs)`."

AbstractGPs.cov(f::SparseFiniteGP) = error(__covariance_error)

AbstractGPs.marginals(f::SparseFiniteGP) = marginals(f.fobs)

AbstractGPs.rand(rng::AbstractRNG, f::SparseFiniteGP, N::Int) = rand(rng, f.fobs, N)
AbstractGPs.rand(f::SparseFiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
AbstractGPs.rand(rng::AbstractRNG, f::SparseFiniteGP) = vec(rand(rng, f, 1))
AbstractGPs.rand(f::SparseFiniteGP) = vec(rand(f, 1))

AbstractGPs.elbo(f::SparseFiniteGP, y::AV{<:Real}) = elbo(VFE(f.finducing), f.fobs, y)

AbstractGPs.logpdf(f::SparseFiniteGP, y::AV{<:Real}) = elbo(VFE(f.finducing), f.fobs, y)

function AbstractGPs.logpdf(f::SparseFiniteGP, Y::AbstractMatrix{<:Real})
    return map(y -> logpdf(f, y), eachcol(Y))
end

function AbstractGPs.posterior(f::SparseFiniteGP, y::AbstractVector{<:Real})
    return posterior(AbstractGPs.VFE(f.finducing), f.fobs, y)
end
