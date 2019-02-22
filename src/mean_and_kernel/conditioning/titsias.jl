"""
    TitsiasKernel <: Kernel

Compute the approximate posterior covariance of a process `g` given pseudo-points `u`.
"""
struct TitsiasKernel{TΣ<:PseudoPointsCov, Tkug<:CrossKernel, Tkgg<:Kernel, Tz<:AV} <: Kernel
    S::TΣ
    kug::Tkug
    kgg::Tkgg
    z::Tz
end

# Binary methods.
function _map(k::TitsiasKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, z, x′)
    return map(k.kgg, x, x′) .- diag_At_B(Ax, Ax′) .+ diag_Xt_invA_Y(Ax, k.S.Λ, Ax′)
end
function _pw(k::TitsiasKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, z, x′)
    return _pw(k.kgg, x, x′) - Ax' * Ax′ + Xt_invA_Y(Ax, k.S.Λ, Ax′)
end

# Unary methods.
function _map(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return map(k.kgg, x) - diag_At_A(Ax) + diag_Xt_invA_X(k.S.Λ, Ax)
end
function _pw(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return pw(k.kgg, x) - Ax'Ax + Xt_invA_X(k.S.Λ, Ax)
end


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
    PseudoPointsCov <: Kernel


"""
struct PseudoPointsCov{TΛ<:Cholesky, TU<:AM} <: Kernel
    Λ::TΛ
    U::TU
end


"""
    ProjKernel <: Kernel

`pw(k, x, x′) = pw(k.k, x, z) * Σ * pw(k.k, z, x′)` where `Σ` is an `M × M` positive
definite matrix given in terms of its Cholesky factorisation `C`, `z` is an `M`-vector. If
`Σ` is a `Real` then it should be positive, and 
"""
struct ProjKernel{TΣ<:PseudoPointsCov, Tk<:CrossKernel, Tz<:AV} <: Kernel
    Σ::TΣ
    k::Tk
    z::Tz
end
function ProjKernel(Σ::PseudoPointsCov, k::CrossKernel, z::Union{Real, AV})
    return ProjKernel(cholesky(Σ), ϕ, _to_vec(z))
end
_to_vec(z::AV) = z
_to_vec(z::Real) = [z]

# Binary methods.
function _map(k::ProjKernel, x::AV, x′::AV)
    C_εf_zx, C_εf_zx′ = k.Σ.U' \ pw(k.k, k.z, x), k.Σ.U' \ pw(k.k, k.z, x′)
    return diag_Xt_invA_Y(C_εf_zx, k.Σ.Λ, C_εf_zx′)
end
function _pw(k::ProjKernel, x::AV, x′::AV)
    C_εf_zx, C_εf_zx′ = k.Σ.U' \ pw(k.k, k.z, x), k.Σ.U' \ pw(k.k, k.z, x′)
    return Xt_invA_Y(C_εf_zx, k.Σ.Λ, C_εf_zx′)
end

# Unary methods.
_map(k::ProjKernel, x::AV) = diag_Xt_invA_X(k.Σ.Λ, k.Σ.U' \ pw(k.k, k.z, x))
_pw(k::ProjKernel, x::AV) = Xt_invA_X(k.Σ.Λ, k.Σ.U' \ pw(k.k, k.z, x))


"""
    ProjCrossKernel <: CrossKernel

`pw(k, x, x′) = pw(k.kl, x, z) * pw(k.kr, z, x′)`.
"""
struct ProjCrossKernel{TΣ<:PseudoPointsCov, Tkl<:CrossKernel, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
    Σ::TΣ
    kl::Tkl
    kr::Tkr
    z::Tz
end

function _map(k::ProjCrossKernel, x::AV, x′::AV)
    return diag_Xt_invA_Y(k.Σ.U' \ pw(k.kl, k.z, x), k.Σ.Λ, k.Σ.U' \ pw(k.kr, k.z, x′))
end
function _pw(k::ProjCrossKernel, x::AV, x′::AV)
    return Xt_invA_Y(k.Σ.U' \ pw(k.kl, k.z, x), k.Σ.Λ, k.Σ.U' \ pw(k.kr, k.z, x′))
end

# Placeholder cross-kernels that don't really do anything
struct LhsProjCrossKernel{TΣ<:PseudoPointsCov, Tkl<:CrossKernel, Tz<:AV} <: CrossKernel
    Σ::TΣ
    kl::Tkl
    z::Tz
end
struct RhsProjCrossKernel{TΣ<:PseudoPointsCov, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
    Σ::TΣ
    kr::Tkr
    z::Tz
end
