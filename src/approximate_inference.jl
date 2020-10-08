import Base: rand, length
import Distributions: logpdf, AbstractMvNormal

export elbo, dtc
export SparseFiniteGP

struct SparseFiniteGP{T1<:FiniteGP, T2<:FiniteGP} <: AbstractMvNormal
    fobs::T1
    finducing::T2
end

length(f::SparseFiniteGP) = length(f.fobs)
mean(f::SparseFiniteGP) = mean(f.fobs)
# in Base, trying to instantiate a large dense matrix through e.g.
# inv(A::SparseArrays.AbstractSparseMatrixCSC) raises an error, since the resulting dense
# matrix will often be enormous and use up all available memory. Doing something similar here.
covariance_error = "The covariance matrix of a sparse GP can often be dense and can cause the computer to run out of memory. If you are sure you have enough memory, you can use `cov(f.fobs)`."
cov(f::SparseFiniteGP) = error(covariance_error) #cov(f.fobs)
# Not sure how to implement the following...
# cov(fx::FiniteGP, gx::FiniteGP) = cov(fx.f, gx.f, fx.x, gx.x)
marginals(f::SparseFiniteGP) = Normal.(mean(f), sqrt.(cov_diag(f.fobs.f, f.fobs.x) .+ diag(f.fobs.Σy)))

rand(rng::AbstractRNG, f::SparseFiniteGP, N::Int) = rand(rng, f.fobs, N)
rand(f::SparseFiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(rng::AbstractRNG, f::SparseFiniteGP) = vec(rand(rng, f, 1))
rand(f::SparseFiniteGP) = vec(rand(f, 1))

elbo(f::SparseFiniteGP, y::AV{<:Real}) = elbo(f.fobs, y, f.finducing)
logpdf(f::SparseFiniteGP, y::AV{<:Real}) = elbo(f, y)
logpdf(f::SparseFiniteGP, Y::AbstractMatrix{<:Real}) = map(y -> logpdf(f, y), eachcol(Y))

PseudoObs(fxu::SparseFiniteGP, y) = PseudoObs(Obs(fxu.fobs, y), fxu.finducing)
←(fxu::SparseFiniteGP, y) = PseudoObs(fxu, y)
