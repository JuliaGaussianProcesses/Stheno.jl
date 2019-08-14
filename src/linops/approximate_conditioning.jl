#
# Computing optimal approximate posterior parameters.
#

function optimal_q(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
    U_y, U = cholesky(Symmetric(f.Σy)).U, cholesky(Symmetric(cov(u))).U
    B_εf, b_y = U' \ (U_y' \ cov(f, u))', U_y' \ (y - mean(f))
    Λ_ε = cholesky(Symmetric(B_εf * B_εf' + I))
    m_ε = Λ_ε \ (B_εf * b_y)
    return m_ε, Λ_ε, U
end
optimal_q(c::Observation, u::FiniteGP) = optimal_q(c.f, c.y, u)

# Sugar for multiple approximate conditioning.
optimal_q(c::Observation, us::Tuple{Vararg{FiniteGP}}) = optimal_q(c, merge(us))
optimal_q(cs::Tuple{Vararg{Observation}}, u::FiniteGP) = optimal_q(merge(cs), u)
function optimal_q(cs::Tuple{Vararg{Observation}}, us::Tuple{Vararg{FiniteGP}})
    return optimal_q(merge(cs), merge(us))
end


# PPGP = "Pseudo-Point GP". Weird internal thing. Please don't use.
struct PPGP{Tm<:AV{<:Real}, TΛ<:Cholesky{<:Real}} <: AbstractGP
    m::Tm
    Λ::TΛ
    gpc::GPC
end


"""
    ApproxObs(u::AbstractGP, z::AV, m::AV{<:Real}, Λ::AM{<:Real}, U::AM)

Construct approximate observations of `u` at `z`, with mean and cov. `m` and `Λ`
"""
struct ApproxObs{Tu<:AbstractGP, Tû<:PPGP, Tz<:AV, TU<:AM, Tα<:AV}
    u::Tu
    û::Tû
    z::Tz
    U::TU
    α::Tα
end

function ApproxObs(u::AbstractGP, z::AV, m::AV, Λ::Cholesky, U::AM)
    return ApproxObs(u, PPGP(m, Λ, u.gpc), z, U, U \ m)
end


"""
    |(f::AbstractGP, ỹ::ApproxObs)

Condition `f` on approximate observations `ỹ`.
"""
|(f::AbstractGP, ỹ::ApproxObs) = CompositeGP((|, f, ỹ), f.gpc)

const approx_cond = Tuple{typeof(|), AbstractGP, ApproxObs}

mean_vector((_, f, ỹ)::approx_cond, x::AV) = mean(f(x)) + cov(f(x), ỹ.u(ỹ.z)) * ỹ.α

function cov((_, f, ỹ)::approx_cond, x::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u(z), f(x))
    return cov(f(x)) - Ax' * Ax + Xt_invA_X(Λ, Ax)
end

function cov((_, f, ỹ)::approx_cond, x::AV, x′::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u(z), f(x))
    Ax′ = U' \ cov(u(z), f(x′))
    return cov(f(x), f(x′)) - Ax' * Ax′ + Xt_invA_Y(Ax, Λ, Ax′)
end

function cov_diag((_, f, ỹ)::approx_cond, x::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u(z), f(x))
    Ax′ = U' \ cov(u(z), f(x′))
    return cov_diag(f(x)) - diag_At_A(Ax) + diag_Xt_invA_X(Ax, Λ, Ax′)
end

function xcov((_, f, ỹ)::approx_cond, f′::PPGP, x::AV, x′::AV)
    throw(error("Not implemented"))
end

function xcov((_, f, ỹ)::approx_cond, f′::AbstractGP, x::AV, x′::AV)
    throw(error("Not implemented"))
end
function xcov(f::AbstractGP, (_, f′, ỹ)::approx_cond, x::AV, x′::AV)
    throw(error("Not implemented"))
end

function xcov_diag((_, f, ỹ)::approx_cond, f′::AbstractGP, x::AV)
    throw(error("Not implemented"))
end
function xcov_diag(f::AbstractGP, (_, f′, ỹ)::approx_cond, x::AV)
    throw(error("Not implemented"))
end

function sample(rng::AbstractRNG, (_, f, ỹ)::approx_cond, x::AV, S::Int)
    throw(error("Not implemented"))
end




# """
#     PseudoPoints

# Some pseudo-points. Really need to improve the documentation here...
# """
# struct PseudoPoints{Tf_q<:GP, Tc<:PseudoPointCache}
#     f_q::Tf_q
#     c::Tc
# end

# function PseudoPoints(c::Observation, u::FiniteGP)
#     m_ε, Λ_ε, U = optimal_q(c.f, c.y, u)
#     return PseudoPoints(u.f, PseudoPointCache(u.x, U, m_ε, Λ_ε))
# end

# PseudoPoints(cs::Tuple{Vararg{Observation}}, u::FiniteGP) = PseudoPoints(merge(cs), u)
# PseudoPoints(c::Observation, us::Tuple{Vararg{FiniteGP}}) = PseudoPoints(c, merge(us))
# function PseudoPoints(cs::Tuple{Vararg{Observation}}, us::Tuple{Vararg{FiniteGP}})
#     return PseudoPoints(merge(cs), merge(us))
# end

# |(f::GP, u::PseudoPoints) = GP(f.gpc, |, f, u)

# function μ_p′(::typeof(|), f::GP, u::PseudoPoints)
#     return ApproxCondMean(u.c, mean(f), kernel(u.f_q, f))
# end

# function k_p′(::typeof(|), f::GP, u::PseudoPoints)
#     return ApproxCondKernel(u.c, kernel(u.f_q, f), kernel(f))
# end

# function k_p′p(::typeof(|), f::GP, u::PseudoPoints, fp::GP)
#     if fp.args[1] isa typeof(|) && fp.args[3] === u
#         f′ = fp.args[2]
#         return ApproxCondCrossKernel(u.c, kernel(u.f_q, f), kernel(u.f_q, f′), kernel(f, f′))
#     else
#         error("Unsupported cross-covariance.")
#     end
# end

# function k_pp′(fp::GP, ::typeof(|) , f::GP, u::PseudoPoints)
#     if fp.args[1] isa typeof(|) && fp.args[3] === u
#         f′ = fp.args[2]
#         return ApproxCondCrossKernel(u.c, kernel(u.f_q, f′), kernel(u.f_q, f), kernel(f′, f))
#     else
#         error("Unsupported cross-covariance")
#     end
# end



# # Sugar.
# function optimal_q(f::AV{<:GP}, y::AV{<:AV{<:Real}}, u::GP, σ::Real)
#     return optimal_q(BlockGP(f), BlockVector(y), u, σ)
# end
# function optimal_q(f::GP, y::AV{<:Real}, u::AV{<:GP}, σ::Real)
#     return optimal_q(f, y, BlockGP(u), σ)
# end
# function optimal_q(f::AV{<:GP}, y::AV{<:AV{<:Real}}, u::AV{<:GP}, σ::Real)
#     return optimal_q(BlockGP(f), BlockVector(y), BlockGP(u), σ)
# end

# """
#     Titsias

# Construct an object which is able to compute an approximate posterior.
# """
# struct Titsias{Tu<:FiniteGP, Tcache<:PseudoPointCache} <: AbstractConditioner
#     u::Tu
#     cache::Tcache
#     function Titsias(u::FiniteGP, m_ε::AV{<:Real}, Λ_ε::Cholesky, U::AbstractMatrix)
#         return Titsias(u, )
#     end
# end
# function Titsias(u::FiniteGP, m_ε::AV{<:Real}, Λ_ε::Cholesky, U::AbstractMatrix)
#     return Titsias()
#     return Titsias(u, m′u, GP(PPC(Λ, U), u.f.gpc))
# end
# Titsias(f::FiniteGP, y::AV{<:Real}, u::FiniteGP) = Titsias(u, optimal_q(f, y, u)...)
# Titsias(c::Observation, u::FiniteGP) = Titsias(c.f, c.y, u)

# # Construct an approximate posterior distribution.
# |(g::GP, c::Titsias) = g | (c.u←c.m′u) + project(kernel(c.u.f, g), c.γ, c.u.x)

# |(g::Tuple{Vararg{GP}}, c::Titsias) = deconstruct(BlockGP([g...]) | c)
# function |(g::BlockGP, c::Titsias)
#     return BlockGP(g.fs .| Ref(c))
# end

# # """
# #     Titsias(
# #         c::Union{Observation, Vector{<:Observation}},
# #         u::Union{GP, Vector{<:GP}},
# #         σ::Real,
# #     )

# # Instantiate the saturated Titsias conditioner.
# # """
# # function Titsias(c::Observation, u::GP, σ::Real)
# #     return Titsias(u, optimal_q(c, u, σ)...)
# # end
# # function Titsias(c::Vector{<:Observation}, u::Vector{<:GP}, σ::Real)
# #     return Titsias(merge(c), BlockGP(u), σ)
# # end
# # function Titsias(c::Observation, u::Vector{<:GP}, σ::Real)
# #     return Titsias(c, BlockGP(u), σ::Real)
# # end
# # function Titsias(c::Vector{<:Observation}, u::GP, σ::Real)
# #     return Titsias(merge(c), u,  σ)
# # end
