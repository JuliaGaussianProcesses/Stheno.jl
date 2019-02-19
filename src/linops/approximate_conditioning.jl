"""
    project(k::CrossKernel, u::GP{<:ZeroMean, <:EagerFinite}, z::AV)


"""
project(k::CrossKernel, u::GP{<:ZeroMean, <:PseudoPointsCov}, z::AV) = GP(project, k, u, z)

μ_p′(::typeof(project), k, u, z) = ZeroMean()
k_p′(::typeof(project), k, u, z) = ProjKernel(u.k, k, z)
function k_p′p(::typeof(project), k, u, z, fp)
    k_ufp = kernel(u, fp)
    if iszero(k_ufp)
        return ZeroKernel()
    elseif u === fp
        return LhsProjCrossKernel(u.k, k, z)
    elseif k_ufp isa RhsProjCrossKernel
        return ProjCrossKernel(u.k, k, k_ufp.kr, z)
    else
        error("k_p′p not defined for $fp")
    end
end
function k_pp′(fp, ::typeof(project), k, u, z)
    k_fpu = kernel(fp, u)
    if iszero(k_fpu)
        return k_fpu
    elseif u === fp 
        return RhsProjCrossKernel(u.k, k, z)
    elseif k_fpu isa LhsProjCrossKernel
        return ProjCrossKernel(u.k, k_fpu.kl, k, z)
    else
        error("k_pp′ not defined for $fp")
    end
end

# Compute the optimal approximate posterior mean and covariance for the Titsias post.
function optimal_q(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    σ = sqrt(FillArrays.getindex_value(f.σ²))
    U = cholesky(cov(u)).U
    Γ = broadcast(/, U' \ cov(u, f), σ)
    Λ = cholesky(Γ * Γ' + I)
    m′u = mean(u) + broadcast(/, U' * (Λ \ (Γ * (y - mean(f)))), σ)
    return m′u, Λ, U
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
end
function Titsias(u::FiniteGP, m′u::AV{<:Real}, Λ, U)
    return Titsias(u, m′u, GP(PseudoPointsCov(Λ, U), u.f.gpc))
end
Titsias(f::FiniteGP, y::AV{<:Real}, u::FiniteGP) = Titsias(u, optimal_q(f, y, u)...)
Titsias(c::Observation, u::FiniteGP) = Titsias(c.f, c.y, u)

# Construct an approximate posterior distribution.
|(g::GP, c::Titsias) = g | (c.u←c.m′u) + project(kernel(c.u.f, g), c.γ, c.u.x)






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
