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

# PPGP = "Pseudo-Point GP". Weird internal thing. Please don't use, or try to access.
# There are a couple of hacks associated with this object. Of particular note is the fact
# that cov(u, z) and cov(u, z, z′) return the precision rather than the cov, while
# cov(u, f, z, x) returns the covariance as expected. This odd behaviour is a hack that the
# routines to compute approximate posterior quantities know how to exploit properly. This
# essentially stems from the lack of a "precision" function, which we need in this case
# sometimes.
struct PPGP{Tm<:AV{<:Real}, TΛ<:Cholesky{<:Real}, Tz<:AV} <: AbstractGP
    m::Tm
    Λ::TΛ
    z::Tz
    n::Int
    gpc::GPC
    function PPGP{Tm, TΛ, Tz}(m::Tm, Λ::TΛ, z::Tz, gpc::GPC) where {Tm, TΛ, Tz}
        gp = new{Tm, TΛ, Tz}(m, Λ, z, next_index(gpc), gpc)
        gpc.n += 1
        return gp
    end
end
function PPGP(m::Tm, Λ::TΛ, z::Tz, gpc::GPC) where {Tm, TΛ, Tz}
    return PPGP{Tm, TΛ, Tz}(m, Λ, z, gpc)
end

mean_vector(u::PPGP, z::AV) = z === u.z ? u.m : throw(ArgumentError("Bad z"))
cov(u::PPGP, z::AV) = z === u.z ? u.Λ : throw(ArgumentError("Bad z"))
cov(u::PPGP, z::AV, z′::AV) = z === z′ ? cov(u, z) : throw(ArgumentError("Bad z"))
function cov(u::PPGP, u′::PPGP, z::AV, z′::AV)
    if u.gpc === u′.gpc && u.n === u′.n
        return cov(u, z, z′)
    else
        throw(ArgumentError("Undefined behaviour"))
    end
end

function cov(u::PPGP, f′::GP, z::AV, x′::AV)
    @assert u.z === z
    return zeros(length(z), length(x′))
end
function cov(f::GP, u′::PPGP, x::AV, z′::AV)
    @assert u′.z === z′
    return zeros(length(x), length(z′))
end

"""
    PseudoObs(u::AbstractGP, z::AV, m::AV{<:Real}, Λ::AM{<:Real}, U::AM)

Construct approximate observations of `u` at `z`, with mean and cov. `m` and `Λ`
"""
struct PseudoObs{Tu<:AbstractGP, Tû<:PPGP, Tz<:AV, TU<:AM, Tα<:AV}
    u::Tu
    û::Tû
    z::Tz
    U::TU
    α::Tα
end

function PseudoObs(u::AbstractGP, z::AV, m::AV, Λ::Cholesky, U::AM)
    return PseudoObs(u, PPGP(m, Λ, z, u.gpc), z, U, U \ m)
end

PseudoObs(c::Observation, u::FiniteGP) = PseudoObs(u.f, u.x, optimal_q(c, u)...)
PseudoObs(c::Observation, us::FiniteGP...) = PseudoObs(c, merge(us))



"""
    |(f::AbstractGP, ỹ::PseudoObs)

Condition `f` on approximate observations `ỹ`.
"""
|(f::AbstractGP, ỹ::PseudoObs) = CompositeGP((|, f, ỹ), f.gpc)

const approx_cond = Tuple{typeof(|), AbstractGP, PseudoObs}

mean_vector((_, f, ỹ)::approx_cond, x::AV) = mean(f(x)) + cov(f(x), ỹ.u(ỹ.z)) * ỹ.α

function cov((_, f, ỹ)::approx_cond, x::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    return cov(f, x) - Ax' * Ax + Xt_invA_X(Λ, Ax)
end
function cov_diag((_, f, ỹ)::approx_cond, x::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    return cov_diag(f, x) - diag_At_A(Ax) + diag_Xt_invA_X(Λ, Ax)
end

function cov((_, f, ỹ)::approx_cond, x::AV, x′::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    Ax′ = U' \ cov(u, f, z, x′)
    return cov(f, x, x′) - Ax' * Ax′ + Xt_invA_Y(Ax, Λ, Ax′)
end
function cov_diag((_, f, ỹ)::approx_cond, x::AV, x′::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    Ax′ = U' \ cov(u, f, z, x′)
    return cov_diag(f, x) - diag_At_B(Ax, Ax′) + diag_Xt_invA_Y(Ax, Λ, Ax′)
end

function cov((_, f, ỹ)::approx_cond, f′::AbstractGP, x::AV, x′::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    Ax′ = U' \ cov(u, f′, z, x′)
    Kzx′ = cov(ỹ.û, f′, z, x′)
    if Kzx′ isa Cholesky
        return cov(f, f′, x, x′) - Ax' * Ax′ + Ax' * (Λ \ Matrix(U))
    else
        return cov(f, f′, x, x′) - Ax' * Ax′ + Xt_invA_Y(Ax, Λ, U' \ Kzx′)
    end
end

function cov(f::AbstractGP, (_, f′, ỹ)::approx_cond, x::AV, x′::AV)
    u, z, U, Λ = ỹ.u, ỹ.z, ỹ.U, ỹ.û.Λ
    Ax = U' \ cov(u, f, z, x)
    Ax′ = U' \ cov(u, f′, z, x′)
    Kzx = cov(ỹ.û, f, z, x)
    if Kzx isa Cholesky
        return cov(f, f′, x, x′) - Ax' * Ax′ + Matrix(U)' * (Λ \ Ax′)
    else
        return cov(f, f′, x, x′) - Ax' * Ax′ + Xt_invA_Y(U' \ Kzx, Λ, Ax′)
    end
end

function cov_diag(args::approx_cond, f′::AbstractGP, x::AV, x′::AV)
    return diag(cov(args, f′, x, x′))
end

function cov_diag(f::AbstractGP, args::approx_cond, x::AV, x′::AV)
    return diag(cov(f, args, x, x′))
end
