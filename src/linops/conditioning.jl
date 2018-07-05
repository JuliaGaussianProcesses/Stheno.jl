import Base: |, merge
export ←, |

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
merge(c::Tuple{Vararg{Observation}}) = BlockGP([get_f.(c)...])←BlockVector([get_y.(c)...])

"""
    |(g::AbstractGP, c::Observation)

Condition `g` on observation `c`.
"""
|(g::GP, c::Observation) = GP(|, g, c.f, CondCache(mean_vec(c.f), cov(c.f), c.y))
|(g::BlockGP, c::Observation) = BlockGP(g.fs .| c)

function μ_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return ConditionalMean(cache, mean(g), kernel(f, g))
end
function k_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return ConditionalKernel(cache, kernel(f, g), kernel(g))
end
function k_p′p(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache, h::AbstractGP)
    return ConditionalCrossKernel(cache, kernel(f, g), kernel(f, h), kernel(g, h))
end
function k_pp′(h::AbstractGP, ::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return ConditionalCrossKernel(cache, kernel(f, h), kernel(f, g), kernel(h, g))
end

# Sugar
|(g::AbstractGP, c::Tuple{Vararg{Observation}}) = g | merge(c)
|(g::Tuple{Vararg{AbstractGP}}, c::Observation) = deconstruct(BlockGP([g...]) | c)
function |(g::Tuple{Vararg{AbstractGP}}, c::Tuple{Vararg{Observation}})
    return deconstruct(BlockGP([g...]) | merge(c))
end

abstract type AbstractConditioner end

"""
    Titsias <: AbstractConditioner

Construct an object which is able to compute an approximate posterior.
"""
struct Titsias{Tu<:AbstractGP, Tm<:AV{<:Real}, Tγ} <: AbstractConditioner
    u::Tu
    m′u::Tm
    γ::Tγ
    function Titsias(u::Tu, m′u::Tm, Σ′uu::AM, gpc::GPC) where {Tu, Tm}
        γ = GP(FiniteKernel(Xtinv_A_Xinv(Σ′uu, cov(u))), gpc)
        return new{Tu, Tm, typeof(γ)}(u, m′u, γ)
    end
end
function |(g::GP, c::Titsias)
    g′ = g | (c.u←c.m′u)
    ĝ = project(kernel(c.u, g), c.γ(eachindex(c.u)))
    return g′ + ĝ
end

function optimal_q(f::AbstractGP, y::AV{<:Real}, u::AbstractGP, σ::Real)
    μᵤ, Σᵤᵤ = mean_vec(u), cov(u)
    U = chol(Σᵤᵤ)
    Γ = (U' \ xcov(u, f)) ./ σ
    Ω, δ = LazyPDMat(Γ * Γ' + I, 0), y - mean_vec(f)
    Σ′ᵤᵤ = Xt_invA_X(Ω, U)
    μ′ᵤ = μᵤ + U' * (Ω \ (Γ * δ)) ./ σ
    return μ′ᵤ, Σ′ᵤᵤ
end

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
    return BlockGP(g.fs .| c)
end

