"""
    CondCache

Cache for use by `CondMean`s, `CondKernel`s and `CondCrossKernel`s.
Avoids recomputing the covariance `Cff` and the Kriging vector `α`.
"""
struct CondCache{TC<:Cholesky, Tα<:AbstractVector{<:Real}, Tx<:AbstractVector}
    C::TC
    α::Tα
    x::Tx
end
function CondCache(k::Kernel, m::MeanFunction, x::AV, y::AV{<:Real}, Σy::AM{<:Real})
    C = cholesky(Symmetric(pw(k, x) + Σy))
    return CondCache(C, C \ (y - map(m, x)), x)
end

"""
    CondMean <: MeanFunction

Represent the posterior mean of `f_p` given observations of `f_q` at `c.x`.

# Fields
- `c::CondCache`: cache of quantities shared by various mean functions and (cross-)kernels
- `mp::MeanFunction`: the (prior) mean function of `f_p`, `mean(f_p)`
- `kqp::CrossKernel`: the cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
"""
struct CondMean{Tc<:CondCache, Tmp<:MeanFunction, Tkqp<:CrossKernel} <: MeanFunction
    c::Tc
    mp::Tmp
    kqp::Tkqp
end
_map(μ::CondMean, x::AV) = bcd(+, _map(μ.mp, x), pw(μ.kqp, μ.c.x, x)' * μ.c.α)


"""
    CondKernel <: Kernel

Represents the posterior covariance of `f_p` given observations of `f_q` at `c.x`.

# Fields
- `c::CondCache`: cache of quantities shared by various mean functions and (cross-)kernels
- `kqp::CrossKernel`: cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
- `kp::Kernel`: kernel of `f_p`, `kernel(f_p)`
"""
struct CondKernel{Tc<:CondCache, Tkqp<:CrossKernel, Tkp<:Kernel} <: Kernel
    c::Tc
    kqp::Tkqp
    kp::Tkp
end

# Binary methods.
function _map(k::CondKernel, x::AV, x′::AV)
    C_qp_x, C_qp_x′ = pw(k.kqp, k.c.x, x), pw(k.kqp, k.c.x, x′)
    return map(k.kp, x, x′) - diag_Xt_invA_Y(C_qp_x, k.c.C, C_qp_x′)
end
function _pw(k::CondKernel, x::AV, x′::AV)
    C_qp_x, C_qp_x′ = pw(k.kqp, k.c.x, x), pw(k.kqp, k.c.x, x′)
    return pw(k.kp, x, x′) - Xt_invA_Y(C_qp_x, k.c.C, C_qp_x′)
end

# Unary methods.
_map(k::CondKernel, x::AV) = map(k.kp, x) - diag_Xt_invA_X(k.c.C, pw(k.kqp, k.c.x, x))
_pw(k::CondKernel, x::AV) = pw(k.kp, x) - Xt_invA_X(k.c.C, pw(k.kqp, k.c.x, x))


"""
    CondCrossKernel <: CrossKernel

Represents the posterior cross-covariance between `f_p` and `f_p′`,given observations of
`f_q` at `c.x`.

# Fields
- `c::CondCache`: cache of quantities shared by various mean functions and (cross-)kernels
- `kqp::CrossKernel`: cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
- `kqp′::CrossKernel`: cross-kernel between `f_q` and `f_p′`, `kernel(f_q, f_p′)`
- `kpp′::CrossKernel`: cross-kernel between `f_p` and f_p′`, `kernel(f_p, f_p′)`
"""
struct CondCrossKernel{
    Tc<:CondCache,
    Tkqp<:CrossKernel,
    Tkqp′<:CrossKernel,
    Tkpp′<:CrossKernel,
} <: CrossKernel
    c::Tc
    kqp::Tkqp
    kqp′::Tkqp′
    kpp′::Tkpp′
end

function _map(k::CondCrossKernel, x::AV, x′::AV)
    C_qp, C_qp′ = pw(k.kqp, k.c.x, x), pw(k.kqp′, k.c.x, x′)
    return map(k.kpp′, x, x′) - diag_Xt_invA_Y(C_qp, k.c.C, C_qp′)
end
function _pw(k::CondCrossKernel, x::AV, x′::AV)
    C_qp, C_qp′ = pw(k.kqp, k.c.x, x), pw(k.kqp′, k.c.x, x′)
    return pw(k.kpp′, x, x′) - Xt_invA_Y(C_qp, k.c.C, C_qp′)
end
