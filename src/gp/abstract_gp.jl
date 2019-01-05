import Base: eachindex, rand
import Statistics: mean, cov
import Distributions: logpdf, ContinuousMultivariateDistribution
export AbstractGP, kernel, rand, logpdf, elbo, marginal_cov, marginal_std, marginals,
    mean_vec, GPC

# A collection of GPs (GPC == "GP Collection"). Used to keep track of internals.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

"""
    AbstractGaussianProcess <: ContinuousMultivariateDistribution

Any subtype of `AbstractGaussianProcess` must define methods of `mean` and `kernel`
"""
abstract type AbstractGaussianProcess <: ContinuousMultivariateDistribution end
const AbstractGP = AbstractGaussianProcess

"""
    mean_vec(f::AbstractGP)

The `Vector` representation of the mean function of `f` if it is finite-dimensional.
"""
mean_vec(f::AbstractGP) = map(mean(f), :)

"""
    cov(f::AbstractGP)

The covariance matrix of `f`, if `f` is finite-dimensional.
"""
cov(f::AbstractGP) = pairwise(kernel(f), :)

"""
    cov(f::AbstractGP, g::AbstractGP)

The covariance matrix between `f` and `g`, if both are finite-dimensional.
"""
cov(f::AbstractGP, g::AbstractGP) = pairwise(kernel(f, g), :, :)

"""
    marginals(f::AbstractGP)

Sugar, returns a vector of Normal distributions representing the marginals of `f`.
"""
marginals(f::AbstractGP) = Normal.(mean_vec(f), sqrt.(map(kernel(f), :)))

"""
    rand(rng::AbstractRNG, f::AbstractGP, N::Int=1)

Obtain `N` independent samples from the GP `f` using `rng`, if `isfinite(length(f))`.
"""
function rand(rng::AbstractRNG, f::AbstractGP, N::Int)
    μ, C = mean_vec(f), cholesky(cov(f) + 1e-6I)
    return μ .+ C.U' * randn(rng, length(μ), N)
end
rand(rng::AbstractRNG, f::AbstractGP) = vec(rand(rng, f, 1))
rand(f::AbstractGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(f::AbstractGP) = vec(rand(f, 1))

"""
    logpdf(f::AbstractGP, y::AbstractVector{<:Real})

The log probability density of `y` under `f`.
"""
function logpdf(f::AbstractGP, y::AbstractVector{<:Real})
    μ, C = mean_vec(f), cholesky(cov(f))
    return -(length(y) * log(2π) + logdet(C) + Xt_invA_X(C, y - μ)) / 2
end

"""
    elbo(f::AbstractGP, y::AbstractVector{<:Real}, u::AbstractGP, σ::Real)

The saturated Titsias-ELBO. Requires a reasonable degree of care.
"""
function elbo(f::AbstractGP, y::AV{<:Real}, u::AbstractGP, σ::Real)
    Γ = (cholesky(cov(u)).U' \ cov(u, f)) ./ σ
    Ω, δ = cholesky(Γ * Γ' + I), y - mean_vec(f)
    return -(length(y) * log(2π * σ^2) + logdet(Ω) - sum(abs2, Γ) +
        (sum(abs2, δ) - sum(abs2, Ω.U' \ (Γ * δ)) + sum(var, marginals(f))) / σ^2) / 2
end
