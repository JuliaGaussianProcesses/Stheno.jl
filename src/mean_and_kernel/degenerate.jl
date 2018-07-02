import Distances: pairwise

"""
    DeltaSumMean{Tϕ, Tg, TZ} <: MeanFunction

"""
struct DeltaSumMean{Tϕ, Tg} <: MeanFunction
    ϕ::Tϕ
    μ::MeanFunction
    g::Tg
end
(μ::DeltaSumMean)(x::Number) = _map(μ, [x])[1]
(μ::DeltaSumMean)(X::AV) = _map(μ, MatData(reshape(X, :, 1)))[1]
function _map(μ::DeltaSumMean, X::AV)
    @show typeof(pairwise(μ.ϕ, :, X)), typeof(eachindex(μ.ϕ, 1))
    @show typeof(μ.ϕ)
    @show typeof(μ.ϕ.ks[1].k), typeof(μ.ϕ.ks[2].k)
    @show μ.ϕ.ks[1].X, X
    @show @which _pairwise(EQ(), μ.ϕ.ks[1].X, X)
    @show size(_pairwise(EQ(), μ.ϕ.ks[1].X, X))
    @show size(_pairwise(EQ(), μ.ϕ.ks[2].X, X))
    @show size(pairwise(μ.ϕ, eachindex(μ.ϕ, 1), X))
    @show @which pairwise(μ.ϕ, eachindex(μ.ϕ, 1), BlockData([X]))
    # @show size(eachindex(μ.ϕ.ks[1], 1)), size(eachindex(μ.ϕ.ks[2], 1))
    # @show size(pairwise(μ.ϕ, :, X))
    # @show size(X)
    return pairwise(μ.ϕ, :, X)' * map(μ.μ, :) + map(μ.g, X)
end
eachindex(μ::DeltaSumMean) = eachindex(g)

"""
    DeltaSumKernel{Tϕ} <: Kernel

"""
struct DeltaSumKernel{Tϕ} <: Kernel
    ϕ::Tϕ
    k::Kernel
end
(k::DeltaSumKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
(k::DeltaSumKernel)(x::Number) = map(k, [x])[1]
function (k::DeltaSumKernel)(x::AV, x′::AV)
    return map(k, MatData(reshape(x, :, 1)), MatData(reshape(x′, :, 1)))[1]
end
(k::DeltaSumKernel)(x::AV) = map(k, MatData(reshape(x, :, 1)))[1]
size(k::DeltaSumKernel, N::Int) = size(k.ϕ, 2)

_map(k::DeltaSumKernel, X::AV) = diag_Xᵀ_A_X(pairwise(k.k, :), pairwise(k.ϕ, :, X))
function _map(k::DeltaSumKernel, X::AV, X′::AV)
    return diag_Xᵀ_A_Y(pairwise(k.ϕ, :, X), pairwise(k.k, :), pairwise(k.ϕ, :, X′))
end
_pairwise(k::DeltaSumKernel, X::AV) = Xt_A_X(pairwise(k.k, :), pairwise(k.ϕ, :, X))
function _pairwise(k::DeltaSumKernel, X::AV, X′::AV)
    return Xt_A_Y(pairwise(k.ϕ, :, X), pairwise(k.k, :), pairwise(k.ϕ, :, X′))
end

"""
    LhsDeltaSumCrossKernel{Tϕ} <: CrossKernel
"""
struct LhsDeltaSumCrossKernel{Tϕ} <: CrossKernel
    ϕ::Tϕ
    k::CrossKernel
end
(k::LhsDeltaSumCrossKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
function (k::LhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return map(k, MatData(reshape(x, : ,1)), MatData(reshape(x′, :, 1)))[1]
end
size(k::LhsDeltaSumCrossKernel, N::Int) = N == 1 ? size(k.ϕ, 2) : size(k.k, 2)

function _map(k::LhsDeltaSumCrossKernel, X::AV, X′::AV)
    return diag_AᵀB(pairwise(k.ϕ, :, X), pairwise(k.k, :, X′))
end
function _pairwise(k::LhsDeltaSumCrossKernel, X::AV, X′::AV)
    return pairwise(k.ϕ, :, X)' * pairwise(k.k, :, X′)
end

"""
    RhsDeltaSumCrossKernel{Tϕ} <: CrossKernel
"""
struct RhsDeltaSumCrossKernel{Tϕ} <: CrossKernel
    k::CrossKernel
    ϕ::Tϕ
end
(k::RhsDeltaSumCrossKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
function (k::RhsDeltaSumCrossKernel)(x::AV, x′::AV)
    return map(k, MatData(reshape(x, :, 1)), MatData(reshape(x′, :, 1)))[1]
end
size(k::RhsDeltaSumCrossKernel, N::Int) = N == 1 ? size(k.k, 2) : size(k.ϕ, 2)

function _map(k::RhsDeltaSumCrossKernel, X::AV, X′::AV)
    return diag_AᵀB(pairwise(k.k, :, X), pairwise(k.ϕ, :, X′))
end
function _pairwise(k::RhsDeltaSumCrossKernel, X::AV, X′::AV)
    return pairwise(k.k, :, X)' * pairwise(k.ϕ, :, X′)
end
