"""
    DeltaSumMean{Tϕ, Tμ<:MeanFunction, Tg} <: MeanFunction

This is an attrociously-named type, apologies. `ϕ` is a callable for which
`pairwise(ϕ, :, x)` does something sensible, `μ` is a finite-dimensional mean function, and
`g` is just some unary function.
"""
struct DeltaSumMean{Tϕ, Tμ<:MeanFunction, Tg} <: MeanFunction
    ϕ::Tϕ
    μ::Tμ
    g::Tg
end
(μ::DeltaSumMean)(x::Number) = _map(μ, [x])[1]
(μ::DeltaSumMean)(X::AV) = _map(μ, ColsAreObs(reshape(X, :, 1)))[1]
_map(μ::DeltaSumMean, x::AV) = bcd(+, pw(μ.ϕ, :, x)' * map(μ.μ, :), _map(μ.g, x))


"""
    DeltaSumKernel{Tϕ, Tk<:Kernel} <: Kernel

`ϕ(:, x)' * pairwise(k, :) * ϕ(:, x′)`
"""
struct DeltaSumKernel{Tϕ, Tk<:Kernel} <: Kernel
    ϕ::Tϕ
    k::Tk
end
(k::DeltaSumKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
(k::DeltaSumKernel)(x::Number) = map(k, [x])[1]
function (k::DeltaSumKernel)(x::AV, x′::AV)
    return map(k, ColsAreObs(reshape(x, :, 1)), ColsAreObs(reshape(x′, :, 1)))[1]
end
(k::DeltaSumKernel)(x::AV) = map(k, ColsAreObs(reshape(x, :, 1)))[1]

_map(k::DeltaSumKernel, x::AV) = diag_Xt_A_X(cholesky(pw(k.k, :)), pw(k.ϕ, :, x))
function _map(k::DeltaSumKernel, x::AV, x′::AV)
    return diag_Xt_A_Y(pw(k.ϕ, :, x), cholesky(pw(k.k, :)), pw(k.ϕ, :, x′))
end
_pw(k::DeltaSumKernel, x::AV) = Xt_A_X(cholesky(pw(k.k, :)), pw(k.ϕ, :, x))
function _pw(k::DeltaSumKernel, x::AV, x′::AV)
    return Xt_A_Y(pw(k.ϕ, :, x), cholesky(pw(k.k, :)), pw(k.ϕ, :, x′))
end


"""
    LhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel} <: CrossKernel
"""
struct LhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel} <: CrossKernel
    ϕ::Tϕ
    k::Tk
end
(k::LhsDeltaSumCrossKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
function (k::LhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return map(k, ColsAreObs(reshape(x, : ,1)), ColsAreObs(reshape(x′, :, 1)))[1]
end

_map(k::LhsDeltaSumCrossKernel, X::AV, X′::AV) = diag_At_B(pw(k.ϕ, :, X), pw(k.k, :, X′))
_pw(k::LhsDeltaSumCrossKernel, X::AV, X′::AV) = pw(k.ϕ, :, X)' * pw(k.k, :, X′)


"""
    RhsDeltaSumCrossKernel{Tϕ} <: CrossKernel
"""
struct RhsDeltaSumCrossKernel{Tϕ, Tk<:CrossKernel} <: CrossKernel
    k::Tk
    ϕ::Tϕ
end
(k::RhsDeltaSumCrossKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
function (k::RhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return map(k, ColsAreObs(reshape(x, :, 1)), ColsAreObs(reshape(x′, :, 1)))[1]
end

_map(k::RhsDeltaSumCrossKernel, X::AV, X′::AV) = diag_At_B(pw(k.k, :, X), pw(k.ϕ, :, X′))
_pw(k::RhsDeltaSumCrossKernel, X::AV, X′::AV) = pw(k.k, :, X)' * pw(k.ϕ, :, X′)
