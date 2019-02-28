const CK = CrossKernel

"""
    PPC <: Kernel

"PPC" === "PsuedoPointsCov"
Placeholder kernel for Titsias implementation.
"""
struct PPC{TΛ<:Cholesky, TU<:UpperTriangular} <: Kernel
    Λ::TΛ
    U::TU
end


struct PseudoPointCache{Tfq, Tm̂ε, TΛ, TU, Tz}
    fq::Tfq
    m̂ε::Tm̂ε
    Λ::TΛ
    U::TU
    z::Tz
end

struct PseudoPointMean{TC<:PseudoPointCache} <: MeanFunction
    c::TC
end
function _map(m::PseudoPointMean, x::AV)
    @assert x === m.c.z || x == m.c.z
    return m.c.m̂ε
end

struct PseudoPointKernel{TC<:PsuedoPointCache} <: Kernel
    c::TC
end
function _pw(c::PseudoPointKernel, x::AV, x′::AV)
    @assert x === c.c.z || x == c.c.z
    @assert x′ === c.c.z || x′ == c.c.z
    return c.c.
end
_pw(c::PseudoPointKernel, x::AV) = _pw(c, x, x)



"""
    TitsiasMean <: MeanFunction

Compute the approximate posterior mean of a process `g` given pseudo-points `u`.
"""
struct TitsiasMean{TΣ<:PPC, Tmg<:MeanFunction, Tkug<:CrossKernel, Tm̂ε<:AV{<:Real}, Tz<:AV} <: MeanFunction
    S::TΣ
    mg::Tmg
    kug::Tkug
    m̂ε::Tm̂ε
    z::Tz
end
_map(m::TitsiasMean, x::AV) = bcd(+, _map(m.mg, x), pw(m.kug, m.z, x)' * (m.S.U \ m.m̂ε))


"""
    TitsiasKernel <: Kernel

Compute the approximate posterior covariance of a process `g` given pseudo-points `u`.
"""
struct TitsiasKernel{TΣ<:PPC, Tkug<:CrossKernel, Tkgg<:Kernel, Tz<:AV} <: Kernel
    S::TΣ
    kug::Tkug
    kgg::Tkgg
    z::Tz
end

# Binary methods.
function _map(k::TitsiasKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, k.z, x′)
    return bcd(-, _map(k.kgg, x, x′), diag_At_B(Ax, Ax′) - diag_Xt_invA_Y(Ax, k.S.Λ, Ax′))
end
function _pw(k::TitsiasKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kug, k.z, x′)
    return bcd(-, _pw(k.kgg, x, x′), Ax' * Ax′ - Xt_invA_Y(Ax, k.S.Λ, Ax′))
end

# Unary methods.
function _map(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return bcd(-, _map(k.kgg, x), diag_At_A(Ax) - diag_Xt_invA_X(k.S.Λ, Ax))
end
function _pw(k::TitsiasKernel, x::AV)
    Ax = k.S.U' \ pw(k.kug, k.z, x)
    return bcd(-, _pw(k.kgg, x), Ax' * Ax - Xt_invA_X(k.S.Λ, Ax))
end


"""
    TitsiasCrossKernel <: CrossKernel

Compute the approximate posterior cross-covariance between `g` and `h` given pseudo-points
`u`.
"""
struct TitsiasCrossKernel{TS<:PPC, Tkug<:CK, Tkuh<:CK, Tkgh<:CK, Tz<:AV{<:Real}
    } <: CrossKernel
    S::TS
    kug::Tkug
    kuh::Tkuh
    kgh::Tkgh
    z::Tz
end

function _map(k::TitsiasCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kuh, k.z, x′)
    return bcd(-, _map(k.kgh, x, x′), diag_At_B(Ax, Ax′) - diag_Xt_invA_Y(Ax, k.S.Λ, Ax′))
end
function _pw(k::TitsiasCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.S.U' \ pw(k.kug, k.z, x), k.S.U' \ pw(k.kuh, k.z, x′)
    return bcd(-, _pw(k.kgh, x, x′), Ax' * Ax′ - Xt_invA_Y(Ax, k.S.Λ, Ax′))
end







# Various (cross-)kernels used to implement the Titsias approximate posterior. Might be
# moved around in the future if we discover a better home for them.


# """
#     ProjKernel <: Kernel

# `pw(k, x, x′) = pw(k.k, x, z) * Σ * pw(k.k, z, x′)` where `Σ` is an `M × M` positive
# definite matrix given in terms of its Cholesky factorisation `C`, `z` is an `M`-vector. If
# `Σ` is a `Real` then it should be positive, and 
# """
# struct ProjKernel{TΣ<:PPC, Tk<:CrossKernel, Tz<:AV} <: Kernel
#     Σ::TΣ
#     k::Tk
#     z::Tz
# end
# function ProjKernel(Σ::PPC, k::CrossKernel, z::Union{Real, AV})
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
# struct ProjCrossKernel{TΣ<:PPC, Tkl<:CrossKernel, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
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
# struct LhsProjCrossKernel{TΣ<:PPC, Tkl<:CrossKernel, Tz<:AV} <: CrossKernel
#     Σ::TΣ
#     kl::Tkl
#     z::Tz
# end
# struct RhsProjCrossKernel{TΣ<:PPC, Tkr<:CrossKernel, Tz<:AV} <: CrossKernel
#     Σ::TΣ
#     kr::Tkr
#     z::Tz
# end
