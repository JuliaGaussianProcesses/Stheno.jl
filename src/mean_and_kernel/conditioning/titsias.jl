const CK = CrossKernel

"""
    PseudoPointsCov <: Kernel

Placeholder kernel for Titsias implementation.
"""
struct PseudoPointsCov{TΛ<:Cholesky, TU<:UpperTriangular} <: Kernel
    Λ::TΛ
    U::TU
end


"""
    TitsiasMean <: MeanFunction
"""
struct TitsiasMean{Tm̂ε<:AV{<:Real}} <: MeanFunction
    m̂ε::Tm̂ε
end

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
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, k.z, x′)
    return bcd(-, _map(k.kgg, x, x′), diag_At_B(Ax, Ax′) + diag_Xt_invA_Y(Ax, k.S.Λ, Ax′))
end
function _pw(k::TitsiasKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, k.z, x′)
    return bcd(-, _pw(k.kgg, x, x′), Ax' * Ax′ + Xt_invA_Y(Ax, k.S.Λ, Ax′))
end

# Unary methods.
function _map(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return bcd(-, _map(k.kgg, x), diag_At_A(Ax) + diag_Xt_invA_X(k.S.Λ, Ax))
end
function _pw(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return bcd(-, _pw(k.kgg, x), Ax' * Ax + Xt_invA_X(k.S.Λ, Ax))
end


"""
    TitsiasCrossKernel <: CrossKernel

Compute the approximate posterior cross-covariance between `g` and `h` given pseudo-points
`u`.
"""
struct TitsiasCrossKernel{TS<:PseudoPointsCov, Tkug<:CK, Tkuh<:CK, Tkgh<:CK, Tz<:AV{<:Real}
    } <: CrossKernel
    S::TS
    kug::Tkug
    kuh::Tkuh
    kgh::Tkgh
    z::Tz
end

function _map(k::TitsiasCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kuh, k.z, x′)
    return bcd(-, _map(k.kgh, x, x′), diag_At_B(Ax, Ax′) + diag_Xt_invA_Y(Ax, k.S.Λ, Ax′))
end
function _pw(k::TitsiasCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kuh, k.z, x′)
    return bcd(-, _pw(k.kgh, x, x′), Ax' * Ax′ + Xt_invA_Y(Ax, k.S.Λ, Ax′))
end







# Various (cross-)kernels used to implement the Titsias approximate posterior. Might be
# moved around in the future if we discover a better home for them.


# """
#     ProjKernel <: Kernel

# `pw(k, x, x′) = pw(k.k, x, z) * Σ * pw(k.k, z, x′)` where `Σ` is an `M × M` positive
# definite matrix given in terms of its Cholesky factorisation `C`, `z` is an `M`-vector. If
# `Σ` is a `Real` then it should be positive, and 
# """
# struct ProjKernel{TΣ<:PseudoPointsCov, Tk<:CrossKernel, Tz<:AV} <: Kernel
#     Σ::TΣ
#     k::Tk
#     z::Tz
# end
# function ProjKernel(Σ::PseudoPointsCov, k::CrossKernel, z::Union{Real, AV})
#     return ProjKernel(cholesky(Σ), ϕ, _to_vec(z))
# end
# _to_vec(z::AV) = z
# _to_vec(z::Real) = [z]

# # Binary methods.
# function _map(k::ProjKernel, x::AV, x′::AV)
#     C_εf_zx, C_εf_zx′ = k.Σ.U' \ pw(k.k, k.z, x), k.Σ.U' \ pw(k.k, k.z, x′)
#     return diag_Xt_invA_Y(C_εf_zx, k.Σ.Λ, C_εf_zx′)
# end
# function _pw(k::ProjKernel, x::AV, x′::AV)
#     C_εf_zx, C_εf_zx′ = k.Σ.U' \ pw(k.k, k.z, x), k.Σ.U' \ pw(k.k, k.z, x′)
#     return Xt_invA_Y(C_εf_zx, k.Σ.Λ, C_εf_zx′)
# end

# # Unary methods.
# _map(k::ProjKernel, x::AV) = diag_Xt_invA_X(k.Σ.Λ, k.Σ.U' \ pw(k.k, k.z, x))
# _pw(k::ProjKernel, x::AV) = Xt_invA_X(k.Σ.Λ, k.Σ.U' \ pw(k.k, k.z, x))


# """
#     ProjCrossKernel <: CrossKernel

# `pw(k, x, x′) = pw(k.kl, x, z) * pw(k.kr, z, x′)`.
# """
# struct ProjCrossKernel{TΣ<:PseudoPointsCov, Tkl<:CrossKernel, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
#     Σ::TΣ
#     kl::Tkl
#     kr::Tkr
#     z::Tz
# end

# function _map(k::ProjCrossKernel, x::AV, x′::AV)
#     return diag_Xt_invA_Y(k.Σ.U' \ pw(k.kl, k.z, x), k.Σ.Λ, k.Σ.U' \ pw(k.kr, k.z, x′))
# end
# function _pw(k::ProjCrossKernel, x::AV, x′::AV)
#     return Xt_invA_Y(k.Σ.U' \ pw(k.kl, k.z, x), k.Σ.Λ, k.Σ.U' \ pw(k.kr, k.z, x′))
# end

# # Placeholder cross-kernels that don't really do anything
# struct LhsProjCrossKernel{TΣ<:PseudoPointsCov, Tkl<:CrossKernel, Tz<:AV} <: CrossKernel
#     Σ::TΣ
#     kl::Tkl
#     z::Tz
# end
# struct RhsProjCrossKernel{TΣ<:PseudoPointsCov, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
#     Σ::TΣ
#     kr::Tkr
#     z::Tz
# end
