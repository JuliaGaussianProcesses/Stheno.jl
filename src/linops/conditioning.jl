import Base: |
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


abstract type AbstractConditioner end

"""
    Titsias <: AbstractConditioner

Construct an object which is able to compute an approximate posterior.
"""
struct Titsias{Tu<:AbstractGP, TZ<:AVM, Tm<:AV{<:Real}, Tγ} <: AbstractConditioner
    u::Tu
    Z::TZ
    m′u::Tm
    γ::Tγ
    function Titsias(u::Tu, Z::TZ, m′u::Tm, Σ′uu::AM, gpc::GPC) where {Tu, TZ, Tm}
        γ = GP(FiniteKernel(Xtinv_A_Xinv(Σ′uu, cov(u, Z))), gpc)
        return new{Tu, TZ, Tm, typeof(γ)}(u, Z, m′u, γ)
    end
end
function |(g::GP, c::Titsias)
    g′ = g | (c.u(c.Z)←c.m′u)
    ϕ = LhsFiniteCrossKernel(kernel(c.u, g), c.Z)
    ĝ = project(ϕ, c.γ, 1:length(c.m′u), ZeroMean{Float64}())
    return return g′ + ĝ
end

function optimal_q(
    f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real},
    u::AV{<:GP}, Z::AV{<:AVM},
    σ::Real,
)
    μᵤ, Σᵤᵤ = mean(u, Z), cov(u, Z)
    U = chol(Σᵤᵤ)
    Γ = (U' \ xcov(u, f, Z, X)) ./ σ
    Ω, δ = LazyPDMat(Γ * Γ' + I, 0), y - mean(f, X)
    Σ′ᵤᵤ = Xt_invA_X(Ω, U)
    μ′ᵤ = μᵤ + (U' * (Ω \ (Γ * δ))) / σ
    return μ′ᵤ, Σ′ᵤᵤ
end
