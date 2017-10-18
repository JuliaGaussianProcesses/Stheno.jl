import Base: mean, cov, chol
export Normal, mean, cov, lpdf, sample, dims

const RealOrVec = Union{Real, AbstractVector{<:Real}}

"""
    Normal{Tμ<:AbstractVector, TΣ<:AbstractPDMat}

Generic multivariate Normal distribution.
"""
struct Normal{Tμ<:RealOrVec, TΣ<:AbstractPDMat}
    μ::Tμ
    Σ::TΣ
    gpc::GPC
    function Normal(μ::T, Σ::V, gpc::GPC) where {T<:AbstractVector, V<:AbstractPDMat}
        (length(μ) != size(Σ, 1) || length(μ) != size(Σ, 2)) &&
            throw(error("μ and Σ are not consistent."))
        return new{T, V}(μ, Σ)
    end
    Normal(μ::T, Σ::V) where {T<:RealOrVec, V<:AbstractPDMat} = new{T, V}(μ, Σ)
end
mean(d::Normal) = d.μ
cov(d::Normal) = d.Σ
dims(d::Normal{<:Real, <:AbstractPDMat}) = size(d.Σ, 1)
dims(d::Normal{<:RealOrVec, <:AbstractPDMat}) = length(d.μ)

"""
    lpdf(d::Normal, x::AbstractVector)

Returns the log probability density of `x` under `d`.
"""
lpdf(d::Normal, x::AbstractVector) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), x - mean(d)))

"""
    sample(rng::AbstractRNG, d::Normal, N::Int=1)

Take `N` samples from `d` using random number generator `rng` (not optional).
"""
sample(rng::AbstractRNG, d::Normal{<:RealOrVec, <:AbstractPDMat}, N::Int=1) =
    mean(d) .+ chol(cov(d)).'randn(rng, dims(d), N)

# """
#     observe(gp::GP, x::Vector, f::Vector{Float64})

# Observe that the value of the `GP` `gp` is `f` at `x`.
# """
# function observe!(gp::GP, x::Vector, f::Vector{Float64})
#     append!(gp.joint.obs, (gp.idx, x, f))
#     return nothing
# end
