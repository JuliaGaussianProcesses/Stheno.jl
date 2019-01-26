"""
    DeltaSumMean{Tϕ, Tμ<:MeanFunction, Tg} <: MeanFunction

This is an attrociously-named type, apologies. `ϕ` is a callable for which
`pairwise(ϕ, :, x)` does something sensible, `μ` is a finite-dimensional mean function, and
`g` is just some unary function.
"""
struct DeltaSumMean{Tϕ, Tμ<:MeanFunction, Tg, Tx} <: MeanFunction
    ϕ::Tϕ
    μ::Tμ
    g::Tg
    x::Tx
end
_map(μ::DeltaSumMean, x::AV) = bcd(+, pw(μ.ϕ, μ.x, x)' * map(μ.μ, μ.x), _map(μ.g, x))


"""
    DeltaSumKernel{Tϕ, Tk<:Kernel} <: Kernel

`ϕ(:, x)' * pairwise(k, :) * ϕ(:, x′)`
"""
struct DeltaSumKernel{Tϕ, Tk<:Kernel, Tx} <: Kernel
    ϕ::Tϕ
    k::Tk
    x::Tx
end
_map(k::DeltaSumKernel, x::AV) = diag_Xt_A_X(cholesky(pw(k.k, k.x)), pw(k.ϕ, k.x, x))
function _map(k::DeltaSumKernel, x::AV, x′::AV)
    return diag_Xt_A_Y(pw(k.ϕ, k.x, x), cholesky(pw(k.k, k.x)), pw(k.ϕ, k.x, x′))
end
_pw(k::DeltaSumKernel, x::AV) = Xt_A_X(cholesky(pw(k.k, k.x)), pw(k.ϕ, k.x, x))
function _pw(k::DeltaSumKernel, x::AV, x′::AV)
    return Xt_A_Y(pw(k.ϕ, k.x, x), cholesky(pw(k.k, k.x)), pw(k.ϕ, k.x, x′))
end


"""
    LhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel} <: CrossKernel
"""
struct LhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel, Tx} <: CrossKernel
    ϕ::Tϕ
    k::Tk
    x::Tx
end
_map(k::LhsDeltaSumCrossKernel, x::AV, x′::AV) = diag_At_B(pw(k.ϕ, k.x, x), pw(k.k, k.x, x′))
_pw(k::LhsDeltaSumCrossKernel, x::AV, x′::AV) = pw(k.ϕ, k.x, x)' * pw(k.k, k.x, x′)


"""
    RhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel} <: CrossKernel
"""
struct RhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel, Tx} <: CrossKernel
    k::Tk
    ϕ::Tϕ
    x::Tx
end
_map(k::RhsDeltaSumCrossKernel, x::AV, x′::AV) = diag_At_B(pw(k.k, k.x, x), pw(k.ϕ, k.x, x′))
_pw(k::RhsDeltaSumCrossKernel, x::AV, x′::AV) = pw(k.k, k.x, x)' * pw(k.ϕ, k.x, x′)
