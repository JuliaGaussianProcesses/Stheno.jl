import Base: |, merge
export ←, |



################################## Exact Conditioning ######################################

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation{Tf<:AbstractGP, Ty<:AbstractVector}
    f::Tf
    y::Ty
end
←(f, y) = Observation(f, y)
get_f(c::Observation) = c.f
get_y(c::Observation) = c.y
function merge(c::Union{AbstractVector{<:Observation}, Tuple{Vararg{Observation}}})
    return BlockGP([get_f.(c)...])←BlockVector([get_y.(c)...])
end

"""
    |(g::AbstractGP, c::Observation)

Condition `g` on observation `c`.
"""
|(g::GP, c::Observation) = GP(|, g, c.f, CondCache(mean_vec(c.f), cov(c.f), c.y))
|(g::BlockGP, c::Observation) = BlockGP(g.fs .| Ref(c))

function μ_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) ?
        mean(g) :
        ConditionalMean(cache, mean(g), kernel(f, g))
end
function k_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) ?
        kernel(g) :
        ConditionalKernel(cache, kernel(f, g), kernel(g))
end
function k_p′p(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache, h::AbstractGP)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(g, h) :
        ConditionalCrossKernel(cache, kernel(f, g), kernel(f, h), kernel(g, h))
end
function k_pp′(h::AbstractGP, ::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(h, g) :
        ConditionalCrossKernel(cache, kernel(f, h), kernel(f, g), kernel(h, g))
end

# Sugar
|(g::AbstractGP, c::Tuple{Vararg{Observation}}) = g | merge(c)
|(g::Tuple{Vararg{AbstractGP}}, c::Observation) = deconstruct(BlockGP([g...]) | c)
function |(g::Tuple{Vararg{AbstractGP}}, c::Tuple{Vararg{Observation}})
    return deconstruct(BlockGP([g...]) | merge(c))
end



################################# Approximate Conditioning #################################

abstract type AbstractConditioner end

FillArrays.Zeros(x::BlockVector) = BlockVector(Zeros.(x.blocks))

"""
    Titsias <: AbstractConditioner

Construct an object which is able to compute an approximate posterior.
"""
struct Titsias{Tu<:AbstractGP, Tm<:AV{<:Real}, Tγ} <: AbstractConditioner
    u::Tu
    m′u::Tm
    γ::Tγ
    function Titsias(u::Tu, m′u::Tm, Σ′uu::AM) where {Tu, Tm}
        μ = EmpiricalMean(Zeros(m′u))
        Σ = EmpiricalKernel(Xtinv_A_Xinv(Σ′uu, cov(u)))
        γ = GP(μ, Σ, u.gpc)
        return new{Tu, Tm, typeof(γ)}(u, m′u, γ)
    end
end
function |(g::GP, c::Titsias)
    g′ = g | (c.u←c.m′u)
    ĝ = project(kernel(c.u, g), c.γ)
    return g′ + ĝ
end

function optimal_q(f::AbstractGP, y::AV{<:Real}, u::AbstractGP, σ::Real)
    μᵤ, Σᵤᵤ = mean_vec(u), cov(u)
    U = cholesky(Σᵤᵤ).U
    Γ = broadcast(/, U' \ xcov(u, f), σ)
    Ω, δ = LazyPDMat(Symmetric(Γ * Γ' + I), 0), y - mean_vec(f)
    Σ′ᵤᵤ = Xt_invA_X(Ω, U)
    μ′ᵤ = μᵤ + broadcast(/, U' * (Ω \ (Γ * δ)), σ)
    return μ′ᵤ, Σ′ᵤᵤ
end
optimal_q(c::Observation, u::AbstractGP, σ::Real) = optimal_q(c.f, c.y, u, σ)

# Sugar.
function optimal_q(f::AV{<:AbstractGP}, y::AV{<:AV{<:Real}}, u::AbstractGP, σ::Real)
    return optimal_q(BlockGP(f), BlockVector(y), u, σ)
end
function optimal_q(f::AbstractGP, y::AV{<:Real}, u::AV{<:AbstractGP}, σ::Real)
    return optimal_q(f, y, BlockGP(u), σ)
end
function optimal_q(f::AV{<:AbstractGP}, y::AV{<:AV{<:Real}}, u::AV{<:AbstractGP}, σ::Real)
    return optimal_q(BlockGP(f), BlockVector(y), BlockGP(u), σ)
end

|(g::Tuple{Vararg{AbstractGP}}, c::Titsias) = deconstruct(BlockGP([g...]) | c)
function |(g::BlockGP, c::Titsias)
    return BlockGP(g.fs .| Ref(c))
end

"""
    Titsias(
        c::Union{Observation, Vector{<:Observation}},
        u::Union{AbstractGP, Vector{<:AbstractGP}},
        σ::Real,
    )

Instantiate the saturated Titsias conditioner.
"""
function Titsias(c::Observation, u::AbstractGP, σ::Real)
    return Titsias(u, optimal_q(c, u, σ)...)
end
function Titsias(c::Vector{<:Observation}, u::Vector{<:AbstractGP}, σ::Real)
    return Titsias(merge(c), BlockGP(u), σ)
end
function Titsias(c::Observation, u::Vector{<:AbstractGP}, σ::Real)
    return Titsias(c, BlockGP(u), σ::Real)
end
function Titsias(c::Vector{<:Observation}, u::AbstractGP, σ::Real)
    return Titsias(merge(c), u,  σ)
end
