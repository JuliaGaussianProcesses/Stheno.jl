import Base: eachindex, rand
import Statistics: mean, cov
import Distributions: logpdf, ContinuousMultivariateDistribution
export AbstractGP, kernel, rand, logpdf, elbo, marginal_cov, marginal_std, marginals,
    mean_vec, xcov, GPC

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
    ==(f::AbstractGP, g::AbstractGP)

Two `AbstractGP`s said to be equal if their marginal distributions are equal. That is, if
the mean function and kernels of both are equal.
"""
==(f::AbstractGP, g::AbstractGP) = (mean(f) == mean(g)) && (kernel(f) == kernel(g))

"""
    length(f::AbstractGP)

The dimensionality of the space over which `f` distributes. May be finite (multivariate
Normal) or infinite (usually, distributes over a function).
"""
length(f::AbstractGP) = length(mean(f))

"""
    eachindex(f::AbstractGP)

Enumerate the dimensions of the space over which `f` distributes if such an enumeration
exists and is finite.
"""
eachindex(f::AbstractGP) = eachindex(mean(f))

"""
    mean_vec(f::AbstractGP)

The `Vector` representation of the mean function of `f`, if `isfinite(length(f))`.
"""
mean_vec(f::AbstractGP) = AbstractVector(mean(f))

"""
    cov(f::AbstractGP)

The covariance matrix of `f`, if `isfinite(length(f))`.
"""
cov(f::AbstractGP) = AbstractMatrix(kernel(f))

"""
    xcov(f::AbstractGP, g::AbstractGP)

The cross-covariance matrix between `f` and `g`, if
`isfinite(length(f)) && isfinite(length(g))`.
"""
xcov(f::AbstractGP, g::AbstractGP) = AbstractMatrix(kernel(f, g))

"""
    marginal_cov(f::AbstractGP)

Efficiently compute of `diag(cov(f))`.
"""
marginal_cov(f::AbstractGP) = map(kernel(f), eachindex(f))

"""
    marginal_std(f::AbstractGP)

Efficiently compute `sqrt.(diag(cov(f)))`.
"""
marginal_std(f::AbstractGP) = sqrt.(marginal_cov(f))

"""
    marginals(f::AbstractGP)

Sugar, equivalent to `(mean(f), marginal_std(f))`.
"""
marginals(f::AbstractGP) = (mean_vec(f), marginal_std(f))

"""
    rand(rng::AbstractRNG, f::AbstractGP, N::Int=1)

Obtain `N` independent samples from the GP `f` using `rng`, if `isfinite(length(f))`.
"""
function rand(rng::AbstractRNG, f::AbstractGP, N::Int)
    return mean_vec(f) .+ chol(cov(f))' * randn(rng, length(f), N)
end
rand(rng::AbstractRNG, f::AbstractGP) = vec(rand(rng, f, 1))
rand(f::AbstractGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(f::AbstractGP) = vec(rand(f, 1))

"""
    logpdf(f::AbstractGP, y::AbstractVector{<:Real})

The log probability density of `y` under `f`.
"""
function logpdf(f::AbstractGP, y::AbstractVector{<:Real})
    μ, Σ = mean_vec(f), cov(f)
    return -0.5 * (length(y) * log(2π) + logdet(Σ) + Xt_invA_X(Σ, y - μ))
end

"""
    elbo(f::AbstractGP, y::AbstractVector{<:Real}, u::AbstractGP, σ::Real)

The saturated Titsias-ELBO. Requires a reasonable degree of care.
"""
function elbo(f::AbstractGP, y::AV{<:Real}, u::AbstractGP, σ::Real)
    Γ = (chol(cov(u))' \ xcov(u, f)) ./ σ
    Ω, δ = LazyPDMat(Γ * Γ' + I, 0), y - mean_vec(f)
    return -0.5 * (length(y) * log(2π * σ^2) + logdet(Ω) - sum(abs2, Γ) +
        (sum(abs2, δ) - sum(abs2, chol(Ω)' \ (Γ * δ)) + sum(marginal_cov(f))) / σ^2)
end
