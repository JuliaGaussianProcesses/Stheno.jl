import Base: rand, length
import Distributions: logpdf, ContinuousMultivariateDistribution

export mean, cov, marginals, rand, logpdf, elbo

"""
    FiniteGP{Tf, Tx}

The finite-dimensional projection of the GP `f` at `x`.
"""
struct FiniteGP{Tf<:AbstractGP, Tx<:AV, TΣy} <: ContinuousMultivariateDistribution
    f::Tf
    x::Tx 
    Σy::TΣy
end
FiniteGP(f::AbstractGP, x::AV, σ²::AV{<:Real}) = FiniteGP(f, x, Diagonal(σ²))
FiniteGP(f::AbstractGP, x::AV, σ²::Real) = FiniteGP(f, x, Fill(σ², length(x)))
FiniteGP(f::AbstractGP, x::AV) = FiniteGP(f, x, 0)

length(f::FiniteGP) = length(f.x)

"""
    mean(f::FiniteGP)

The mean vector of `f`.
"""
mean(f::FiniteGP) = map(mean(f.f), f.x)

"""
    cov(f::FiniteGP)

The covariance matrix of `f`.
"""
cov(f::FiniteGP) = pairwise(kernel(f.f), f.x) + f.Σy

"""
    cov(f::FiniteGP, g::FiniteGP)

The cross-covariance between `f` and `g`.
"""
cov(f::FiniteGP, g::FiniteGP) = pairwise(kernel(f.f, g.f), f.x, g.x)

"""
    marginals(f::FiniteGP)

Sugar, returns a vector of Normal distributions representing the marginals of `f`.
"""
marginals(f::FiniteGP) = Normal.(mean(f), sqrt.(map(kernel(f.f), f.x) .+ diag(f.Σy)))

"""
    rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

Obtain `N` independent samples from the GP `f` using `rng`.
"""
function rand(rng::AbstractRNG, f::FiniteGP, N::Int)
    Σ = cov(f)
    μ, C = mean(f), cholesky(Symmetric(Σ))
    return μ .+ C.U' * _rand_eps(rng, Σ, N)
end

rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)

_rand_eps(rng::AbstractRNG, Σ::AbstractMatrix, N::Int) = randn(rng, size(Σ, 2), N)

function _rand_eps(rng::AbstractRNG, Σ::AbstractBlockMatrix, N::Int)
    return BlockMatrix([randn(rng, sz, N) for sz in blocksizes(Σ, 2), N in [N]])
end

function rand(rng::AbstractRNG, f::FiniteGP)
    Σ = cov(f)
    μ, C = mean(f), cholesky(Symmetric(Σ))
    return μ .+ C.U' * _rand_eps(rng, Σ)
end

rand(f::FiniteGP) = vec(rand(f, 1))

_rand_eps(rng::AbstractRNG, Σ::AbstractMatrix) = randn(rng, size(Σ, 2))

function _rand_eps(rng::AbstractRNG, Σ::AbstractBlockMatrix)
    return BlockVector([randn(rng, sz) for sz in blocksizes(Σ, 2)])
end

"""
    logpdf(f::FiniteGP, y::AbstractVector{<:Real})

The log probability density of `y` under `f`.
"""
function logpdf(f::FiniteGP, y::AbstractVector{<:Real})
    μ, C = mean(f), cholesky(Symmetric(cov(f)))
    return -(length(y) * log(2π) + logdet(C) + Xt_invA_X(C, y - μ)) / 2
end

"""
   elbo(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)

The saturated Titsias-ELBO.
"""
function elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    @assert length(f) == length(y)
    chol_Σy = cholesky(Symmetric(f.Σy))

    # A = (cholesky(Symmetric(cov(u))).U' \ cov(u, f)) / chol_Σy.U
    A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
    Λ_ε, δ = cholesky(Symmetric(A * A' + I)), chol_Σy.U' \ (y - mean(f))

    return -(length(y) * log(2π) + logdet(chol_Σy) + logdet(Λ_ε) +
        sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ)) +
        tr_Cf_invΣy(f, f.Σy, chol_Σy) - sum(abs2, A)) / 2
end

# Compute tr(Cf / Σy) efficiently for different types of Σy. For dense Σy you obviously need
# to compute the entirety of Cf, which is bad, but for particular structured Σy one requires
# only a subset of the elements. Σy isa UniformScaling is version usually considered.
function tr_Cf_invΣy(f::FiniteGP, Σy::UniformScaling, chol_Σy::Cholesky)
    return sum(map(kernel(f.f), f.x)) / Σy.λ
end
function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal, chol_Σy::Cholesky)
    return sum(map(kernel(f.f), f.x) ./ diag(Σy))
end

# """
#     elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP, mε::AV{<:Real}, Λε::AM{<:Real})

# The unsaturated Titsias-ELBO.
# """
# function elbo(f::FiniteGP, y::AV{<:Real}, u::FiniteGP, mε::AV{<:Real}, Λε::AM{<:Real})
#     @assert length(u.x) == length(mε)
#     @assert size(Λε) == (length(mε), length(mε))
#     # do stuff.
# end


#
# `logpdf` and `rand` for collections of processes #
#

# function rand(rng::AbstractRNG, f::BlockGP, N::Int)
#     M = BlockArray(undef_blocks, AbstractMatrix{Float64}, length.(f.fs), [N])
#     μ = mean(f)
#     for b in eachindex(f.fs)
#         setblock!(M, getblock(μ, b) * ones(1, N), b, 1)
#     end
#     return M + chol(cov(f))' * BlockMatrix(randn.(Ref(rng), length.(f.fs), N))
# end
# rand(f::BlockGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)

# function rand(rng::AbstractRNG, f::BlockGP)
#     return mean(f) + chol(cov(f))' * BlockVector(randn.(Ref(rng), length.(f.fs)))
# end
# rand(f::BlockGP) = rand(Random.GLOBAL_RNG, f)

# # Convenience methods for invoking `logpdf` and `rand` with multiple processes.
# logpdf(fs::AV{<:AbstractGP}, ys::AV{<:AV{<:Real}}) = logpdf(BlockGP(fs), BlockVector(ys))

function finites_to_block(fs::AV{<:FiniteGP})
    Σys = map(f->f.Σy, fs)
    sizes = map(Σy->size(Σy, 1), Σys)
    return FiniteGP(
        BlockGP(map(f->f.f, fs)),
        BlockData(map(f->f.x, fs)),
        Matrix(_BlockArray(Diagonal(Σys), sizes, sizes)),
    )
end

function rand(rng::AbstractRNG, fs::AV{<:FiniteGP}, N::Int)
    Y = rand(rng, finites_to_block(fs), N)
    sz = cumsum(map(length, fs))
    return [Y[sz[n]-length(fs[n])+1:sz[n], :] for n in eachindex(fs)]
end
rand(rng::AbstractRNG, fs::AV{<:FiniteGP}) = vec.(rand(rng, fs, 1))
rand(fs::AV{<:FiniteGP}, N::Int) = rand(Random.GLOBAL_RNG, fs, N)
rand(fs::AV{<:FiniteGP}) = vec.(rand(Random.GLOBAL_RNG, fs))

logpdf(fs::AV{<:FiniteGP}, ys::AV{<:AV{<:Real}}) = logpdf(finites_to_block(fs), vcat(ys...))
