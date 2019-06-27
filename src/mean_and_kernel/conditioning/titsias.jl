"""
    PseudoPointCache

Various quantities required for approximate conditioning. We let
`ε := U' \\ (f_q(z) - mean(f_q(z)))` be normally distributed with mean `m̂ε` and precision
`Λ`.

# Fields
- `z::AbstractVector`: the locations of the pseudo-points in `f_q`
- `U::UpperTriangular`: the upper-triangular cholesky factor `cholesky(pw(fq, z)).U`
- `m̂ε::AbstractVector{<:Real}`: the mean of `ε`
- `Λ::Cholesky`: the precision of `ε`
"""
struct PseudoPointCache{Tz<:AV, TU<:UpperTriangular, Tm̂ε<:AV{<:Real}, TΛ<:Cholesky}
    z::Tz
    U::TU
    m̂ε::Tm̂ε
    Λ::TΛ
end
const PPC = PseudoPointCache


"""
    ApproxCondMean <: MeanFunction

Represents the approximate posterior mean function of the pth process `f_p` in a programme,
conditioned on approximate observations of the qth process `f_q`.

# Fields
- `c::PPC`: cache storing various quantities shared between different means / kernels
- `mp::MeanFunction`: the mean function of `f_p`, `mean(f_p)`
- `kqp::CrossKernel`: the cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
"""
struct ApproxCondMean{Tc<:PPC, Tmp<:MeanFunction, Tkqp<:CrossKernel} <: MeanFunction
    c::Tc
    mp::Tmp
    kqp::Tkqp
end
ew(m::ApproxCondMean, x::AV) = ew(m.mp, x) .+ pw(m.kqp, m.c.z, x)' * (m.c.U \ m.c.m̂ε)


"""
    ApproxCondKernel <: Kernel

Represents the approximate posterior covariance of a the pth process `f_p` in a programme
conditioned on approximate observations of the qth process `f_q`.

# Fields
- `c::PPC`: cache storing various quanties shared between different means / kernels
- `kqp::CrossKernel`: the cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
- `kp::Kernel`: the kernel of `f_p`, `kernel(f_p)`
"""
struct ApproxCondKernel{Tc<:PPC, Tkqp<:CrossKernel, Tkp<:Kernel} <: Kernel
    c::Tc
    kqp::Tkqp
    kp::Tkp
end

# Binary methods.
function ew(k::ApproxCondKernel, x::AV, x′::AV)
    Ax, Ax′ = k.c.U' \ pw(k.kqp, k.c.z, x), k.c.U' \ pw(k.kqp, k.c.z, x′)
    return ew(k.kp, x, x′) .- diag_At_B(Ax, Ax′) .+ diag_Xt_invA_Y(Ax, k.c.Λ, Ax′)
end
function pw(k::ApproxCondKernel, x::AV, x′::AV)
    Ax, Ax′ = k.c.U' \ pw(k.kqp, k.c.z, x), k.c.U' \ pw(k.kqp, k.c.z, x′)
    return pw(k.kp, x, x′) .- Ax' * Ax′ .+ Xt_invA_Y(Ax, k.c.Λ, Ax′)
end

# Unary methods.
function ew(k::ApproxCondKernel, x::AV)
    Ax = k.c.U' \ pw(k.kqp, k.c.z, x)
    return ew(k.kp, x) .- diag_At_A(Ax) .+ diag_Xt_invA_X(k.c.Λ, Ax)
end
function pw(k::ApproxCondKernel, x::AV)
    Ax = k.c.U' \ pw(k.kqp, k.c.z, x)
    return pw(k.kp, x) .- Ax' * Ax .+ Xt_invA_X(k.c.Λ, Ax)
end


"""
    ApproxCondCrossKernel <: CrossKernel

Represents the approximate posterior cross-covariance between the pth and p′th processes
`f_p` and `f_p′` in a programme, given approximate observations of the qth process `f_q`.

# Fields
- `c::PPC`: cache storing various quanties shared between different means / kernels
- `kqp::CrossKernel`: the cross-kernel between `f_q` and `f_p`, `kernel(f_q, f_p)`
- `kqp′::CrossKernel`: the cross-kernel between `f_q` and `f_p′`, `kernel(f_q, f_p′)`
- `kpp′::CrossKernel`: the cross-kernel between `f_p` and `f_p′`, `kernel(f_p, f_p′)`
"""
struct ApproxCondCrossKernel{
    Tc<:PPC,
    Tkqp<:CrossKernel,
    Tkqp′<:CrossKernel,
    Tkpp′<:CrossKernel,
} <: CrossKernel
    c::Tc
    kqp::Tkqp
    kqp′::Tkqp′
    kpp′::Tkpp′
end

function ew(k::ApproxCondCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.c.U' \ pw(k.kqp, k.c.z, x), k.c.U' \ pw(k.kqp′, k.c.z, x′)
    return ew(k.kpp′, x, x′) .- diag_At_B(Ax, Ax′) .+ diag_Xt_invA_Y(Ax, k.c.Λ, Ax′)
end
function pw(k::ApproxCondCrossKernel, x::AV, x′::AV)
    Ax, Ax′ = k.c.U' \ pw(k.kqp, k.c.z, x), k.c.U' \ pw(k.kqp′, k.c.z, x′)
    return pw(k.kpp′, x, x′) .- Ax' * Ax′ .+ Xt_invA_Y(Ax, k.c.Λ, Ax′)
end
