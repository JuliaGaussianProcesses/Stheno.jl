"""
    project(k::CrossKernel, u::GP{<:ZeroMean, <:EagerFinite}, z::AV)


"""
project(k::CrossKernel, u::GP{<:ZeroMean, <:EagerFinite}, z::AV) = GP(project, k, u, z)

μ_p′(::typeof(project), k, u, z) = ZeroMean()
k_p′(::typeof(project), k, u, z) = ProjKernel(cholesky(u.k.Σ), k, z)
function k_p′p(::typeof(project), k, u, z, fp)
    k_ufp = kernel(u, fp)
    return iszero(k_ufp) ? ZeroKernel() : ProjCrossKernel(k, z, k_ufp)
end
function k_pp′(fp, ::typeof(project), k, u, z)
    k_fpu = kernel(fp, u)
    return iszero(k_fpu) ? ZeroKernel() : ProjCrossKernel(k_fpu, z, k)
end

# Compute the optimal approximate posterior mean and covariance for the Titsias post.
function optimal_q(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    σ = sqrt(FillArrays.getindex_value(f.σ²))
    μᵤ, Σᵤᵤ = mean(u), cov(u)
    U = cholesky(Σᵤᵤ).U
    Γ = broadcast(/, U' \ cov(u, f), σ)
    Ω = cholesky(Γ * Γ' + I)
    Σ′ᵤᵤ = Xt_invA_X(Ω, U)
    μ′ᵤ = μᵤ + broadcast(/, U' * (Ω \ (Γ * (y - mean(f)))), σ)
    return μ′ᵤ, Σ′ᵤᵤ
end
optimal_q(c::Observation, u::FiniteGP) = optimal_q(c.f, c.y, u)

abstract type AbstractConditioner end

"""
    Titsias <: AbstractConditioner

Construct an object which is able to compute an approximate posterior.
"""
struct Titsias{Tu<:FiniteGP, Tm<:AV{<:Real}, Tγ} <: AbstractConditioner
    u::Tu
    m′u::Tm
    γ::Tγ
    function Titsias(u::Tu, m′u::Tm, Σ′uu::AM) where {Tu<:FiniteGP, Tm}
        Σ = EagerFinite(Xtinv_A_Xinv(cholesky(Σ′uu), cholesky(cov(u))))
        γ = GP(Σ, u.f.gpc)
        return new{typeof(u), Tm, typeof(γ)}(u, m′u, γ)
    end
end
function |(g::GP, c::Titsias)
    g′ = g | (c.u←c.m′u)
    ĝ = project(kernel(c.u.f, g), c.γ, c.u.x)
    return g′ + ĝ
end




# FillArrays.Zeros(x::BlockVector) = BlockVector(Zeros.(x.blocks))

# # Sugar.
# function optimal_q(f::AV{<:AbstractGP}, y::AV{<:AV{<:Real}}, u::AbstractGP, σ::Real)
#     return optimal_q(BlockGP(f), BlockVector(y), u, σ)
# end
# function optimal_q(f::AbstractGP, y::AV{<:Real}, u::AV{<:AbstractGP}, σ::Real)
#     return optimal_q(f, y, BlockGP(u), σ)
# end
# function optimal_q(f::AV{<:AbstractGP}, y::AV{<:AV{<:Real}}, u::AV{<:AbstractGP}, σ::Real)
#     return optimal_q(BlockGP(f), BlockVector(y), BlockGP(u), σ)
# end

# |(g::Tuple{Vararg{AbstractGP}}, c::Titsias) = deconstruct(BlockGP([g...]) | c)
# function |(g::BlockGP, c::Titsias)
#     return BlockGP(g.fs .| Ref(c))
# end

# """
#     Titsias(
#         c::Union{Observation, Vector{<:Observation}},
#         u::Union{AbstractGP, Vector{<:AbstractGP}},
#         σ::Real,
#     )

# Instantiate the saturated Titsias conditioner.
# """
# function Titsias(c::Observation, u::AbstractGP, σ::Real)
#     return Titsias(u, optimal_q(c, u, σ)...)
# end
# function Titsias(c::Vector{<:Observation}, u::Vector{<:AbstractGP}, σ::Real)
#     return Titsias(merge(c), BlockGP(u), σ)
# end
# function Titsias(c::Observation, u::Vector{<:AbstractGP}, σ::Real)
#     return Titsias(c, BlockGP(u), σ::Real)
# end
# function Titsias(c::Vector{<:Observation}, u::AbstractGP, σ::Real)
#     return Titsias(merge(c), u,  σ)
# end
