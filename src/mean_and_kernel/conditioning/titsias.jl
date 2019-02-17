# """
#     TitsiasMean <: MeanFunction

# The approximate posterior mean of a process `g` given pseudo-points `u`.
# """
# struct TitsiasMean{Tm<:CondMean} <: MeanFunction
#     m::Tm
# end
# TitsiasMean(cache::CondCache, kug, mg) = TitsiasMean(CondMean(cache, mg, kug))

# _map(m::TitsiasMean, x::AV) = _map(m, x)


# """
#     TitsiasKernel <: Kernel

# # Compute the approximate posterior covariance of a process `g` given pseudo-points `u`.
# # """
# struct TitsiasKernel{TS, Tck<:CondKernel} <: Kernel
#     S::TS
#     ck::Tck
# end
# TitsiasKernel(cache::CondCache, kug, kgg, S) = TitsiasKernel(S, CondKernel(cache, kug, kgg))
# getproperty(k::TitsiasKernel, d::Symbol) =
#     d === :kug ? k.ck.kfg :
#     d === :z ? k.ck.c.x :
#     getfield(k, d)

# # Binary methods.
# function _map(k::TitsiasKernel, x::AV, x′::AV)
#     Σug_x, Σug_x′ = pw(k.kug, k.z, x), pw(k.kug, k.z, x′)
#     return bcd(+, _map(k.ck, x, x′), diag_Xt_A_Y(Σug_x, k.S, Σug_x′))
# end
# function _pw(k::TitsiasKernel, x::AV, x′::AV)
#     return bcd(+, _pw(k.ck, x, x′), Xt_A_Y(pw(k.kug, k.z, x), k.S, pw(k.kug, k.z, x′)))
# end

# # Unary methods.
# _map(k::TitsiasKernel, x::AV) = bcd(+, _map(k.ck, x), diag_Xt_A_X(k.S, pw(k.kug, k.z, x)))
# _pw(k::TitsiasKernel, x::AV) = bcd(+, _pw(k.ck, x), Xt_A_X(k.S, pw(k.kug, k.z, x)))


# """
#     TitsiasCrossKernel <: CrossKernel

# Compute the approximate posterior cross-covariance between `g` and `h` given pseudo-points
# `u`.
# """
# struct TitsiasCrossKernel{TS, Tck<:CondCrossKernel} <: Kernel
#     S::TS
#     ck::Tck
# end
# function TitsiasCrossKernel(cache::CondCache, kug, kuh, kgh, S)
#     return TitsiasCrossKernel(S, CondCrossKernel(cache, kug, kuh, kgh))
# end
# getproperty(k::TitsiasCrossKernel, d::Symbol) =
#     d === :kug ? k.ck.kfg :
#     d === :kuh ? k.ck.kfh :
#     d === :z ? k.ck.c.x :
#     getfield(k, d)

# function _map(k::TitsiasCrossKernel, x::AV, x′::AV)
#     Σug_zx, Σuh_zx′ = pw(k.kug, k.z, x), pw(k.kuh, k.z, x′)
#     return bcd(+, _map(k.ck, x, x′), diag_Xt_A_Y(Σug_zx, k.S, Σuh_zx′))
# end
# function _pw(k::TitsiasCrossKernel, x::AV, x′::AV)
#     return bcd(+, _pw(k.ck, x, x′), Xt_A_Y(pw(k.kug, k.z, x), k.S, pw(k.kuh, k.z, x′)))
# end







# Various (cross-)kernels used to implement the Titsias approximate posterior. Might be
# moved around in the future if we discover a better home for them.

"""
    EagerFinite <: Kernel

A kernel whose domain is finite, and is represented eagerly in terms of the Cholesky
factorisation of its covariance matrix. Not a real kernel as you can't do a lot with it.
Consequently, you won't be able to do a lot with the process defined in terms of this
kernel.
"""
struct EagerFinite{TΣ<:AbstractMatrix} <: Kernel
    Σ::TΣ
end

function pw(k::EagerFinite, x::AV, x′::AV)
    @assert length(x) == length(x′)
    @assert length(x) == size(k.Σ, 1)
    return k.Σ
end


"""
    ProjKernel <: Kernel

`pw(k, x, x′) = pw(k.k, x, z) * Σ * pw(k.k, z, x′)` where `Σ` is an `M × M` positive
definite matrix given in terms of its Cholesky factorisation `C`, `z` is an `M`-vector. If
`Σ` is a `Real` then it should be positive, and 
"""
struct ProjKernel{TC<:Cholesky, Tk<:CrossKernel, Tz<:AV} <: Kernel
    C::TC
    k::Tk
    z::Tz
end
function ProjKernel(Σ::Union{Real, AbstractMatrix}, k::CrossKernel, z::Union{Real, AV})
    return ProjKernel(cholesky(Σ), ϕ, _to_vec(z))
end
_to_vec(z::AV) = z
_to_vec(z::Real) = [z]

# Binary methods.
_map(k::ProjKernel, x::AV, x′::AV) = diag_Xt_A_Y(pw(k.k, k.z, x), k.C, pw(k.k, k.z, x′))
_pw(k::ProjKernel, x::AV, x′::AV) = Xt_A_Y(pw(k.k, k.z, x), k.C, pw(k.k, k.z, x′))

# Unary methods.
_map(k::ProjKernel, x::AV) = diag_Xt_A_X(k.C, pw(k.k, k.z, x))
_pw(k::ProjKernel, x::AV) = Xt_A_X(k.C, pw(k.k, k.z, x))


"""
    ProjCrossKernel <: CrossKernel

`pw(k, x, x′) = pw(k.kl, x, z) * pw(k.kr, z, x′)`.
"""
struct ProjCrossKernel{Tkl<:CrossKernel, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
    kl::Tkl
    z::Tz
    kr::Tkr
end

_map(k::ProjCrossKernel, x::AV, x′::AV) = diag_At_B(pw(k.kl, x, k.z)', pw(k.kr, k.z, x′))
_pw(k::ProjCrossKernel, x::AV, x′::AV) = pw(k.kl, x, k.z) * pw(k.kr, k.z, x′)
