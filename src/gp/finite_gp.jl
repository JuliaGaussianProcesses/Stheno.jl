import Base: rand
import Distributions: logpdf, ContinuousMultivariateDistribution

export mean, cov, marginals, rand, logpdf, elbo

"""
    FiniteGP{Tf, Tx}

The finite-dimensional projection of the GP `f` at `x`.
"""
struct FiniteGP{Tf<:AbstractGP, Tx} <: ContinuousMultivariateDistribution
    f::Tf
    x::Tx 
end

"""
    mean(f::FiniteGP)

The mean vector of `f`.
"""
mean(f::FiniteGP) = map(mean(f.f), f.x)

"""
    cov(f::FiniteGP)

The covariance matrix of `f`.
"""
cov(f::FiniteGP) = pairwise(kernel(f.f), f.x)

"""
    cov(f::FiniteGP, g::FiniteGP)

The cross-covariance between `f` and `g`.
"""
cov(f::FiniteGP, g::FiniteGP) = pairwise(kernel(f.f, g.f), f.x, g.x)

"""
    marginals(f::FiniteGP)

Sugar, returns a vector of Normal distributions representing the marginals of `f`.
"""
marginals(f::FiniteGP) = Normal.(mean(f), sqrt.(map(kernel(f.f), f.x)))

"""
    rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

Obtain `N` independent samples from the GP `f` using `rng`.
"""
function rand(rng::AbstractRNG, f::FiniteGP, N::Int)
    μ, C = mean(f), cholesky(cov(f) + 1e-6I)
    return μ .+ C.U' * randn(rng, length(μ), N)
end
rand(rng::AbstractRNG, f::FiniteGP) = vec(rand(rng, f, 1))
rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
rand(f::FiniteGP) = vec(rand(f, 1))

"""
    logpdf(f::FiniteGP, y::AbstractVector{<:Real})

The log probability density of `y` under `f`.
"""
function logpdf(f::FiniteGP, y::AbstractVector{<:Real})
    μ, C = mean(f), cholesky(cov(f))
    return -(length(y) * log(2π) + logdet(C) + Xt_invA_X(C, y - μ)) / 2
end

"""
    elbo(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP, σ::Real)

The saturated Titsias-ELBO. Requires a reasonable degree of care.
"""
function elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP, σ::Real)
    Γ = (cholesky(cov(u)).U' \ cov(u, f)) ./ σ
    Ω, δ = cholesky(Γ * Γ' + I), y - mean(f)
    return -(length(y) * log(2π * σ^2) + logdet(Ω) - sum(abs2, Γ) +
        (sum(abs2, δ) - sum(abs2, Ω.U' \ (Γ * δ)) + sum(var, marginals(f))) / σ^2) / 2
end