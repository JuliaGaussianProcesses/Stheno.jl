import Base: rand, length
import Distributions: logpdf, AbstractMvNormal

export elbo, dtc
export SparseFiniteGP

"""
    SparseFiniteGP{T1<:FiniteGP, T2<:FiniteGP}

A finite-dimensional projection of an `AbstractGP` `f` at locations `x`, which uses a second
`FiniteGP` defined at a sparse set of inducing points [1] to do approximate inference.

This object has similar methods to an ordinary `FiniteGP`, but note that the methods for
`logpdf` and `←` are just convenience wrappers around `elbo` and `PseudoObs`.

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.

```jldoctest
julia> f = wrap(GP(Matern32Kernel()), GPC());

julia> fobs = f(rand(100));

julia> finducing = f(0:0.1:1);

julia> fxu = SparseFiniteGP(fobs, finducing);

julia> y = rand(fxu);

julia> logpdf(fxu, y) < logpdf(fobs, y)
true

julia> fpost = f | (fxu ← y);
```
"""
struct SparseFiniteGP{T1<:FiniteGP, T2<:FiniteGP} <: AbstractMvNormal
    fobs::T1
    finducing::T2
end

length(f::SparseFiniteGP) = length(f.fobs)
mean(f::SparseFiniteGP) = mean(f.fobs)
covariance_error = "The covariance matrix of a sparse GP can often be dense and " *
    "can cause the computer to run out of memory. If you are sure you have enough " *
    "memory, you can use `cov(f.fobs)`."
cov(f::SparseFiniteGP) = error(covariance_error) #cov(f.fobs)
# Not sure how to implement the following...
# cov(fx::FiniteGP, gx::FiniteGP) = cov(fx.f, gx.f, fx.x, gx.x)
marginals(f::SparseFiniteGP) = marginals(f.fobs)

rand(rng::AbstractRNG, f::SparseFiniteGP, N::Int) = rand(rng, f.fobs, N)
rand(f::SparseFiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(rng::AbstractRNG, f::SparseFiniteGP) = vec(rand(rng, f, 1))
rand(f::SparseFiniteGP) = vec(rand(f, 1))

elbo(f::SparseFiniteGP, y::AV{<:Real}) = elbo(f.fobs, y, f.finducing)
logpdf(f::SparseFiniteGP, y::AV{<:Real}) = elbo(f, y)
logpdf(f::SparseFiniteGP, Y::AbstractMatrix{<:Real}) = map(y -> logpdf(f, y), eachcol(Y))

PseudoObs(fxu::SparseFiniteGP, y) = PseudoObs(Obs(fxu.fobs, y), fxu.finducing)
←(fxu::SparseFiniteGP, y) = PseudoObs(fxu, y)
