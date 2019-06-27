import Distances: pairwise
import Base: eachindex

"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache
    Σff::LazyPDMat
    α::AV{<:Real}
    function CondCache(mf::AV{<:Real}, Σff::AM, f::AV{<:Real})
        return new(Σff, Σff \ (f - mf))
    end
end
function CondCache(kff::Kernel, μf::MeanFunction, X::AV, f::AV{<:Real})
    return CondCache(map(μf, X), pairwise(kff, X), f)
end
show(io::IO, ::CondCache) = show(io, "CondCache")

"""
    ConditionalMean <: MeanFunction

Computes the mean of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalMean <: MeanFunction
    c::CondCache
    μg::MeanFunction
    kfg::CrossKernel
end
length(μ::ConditionalMean) = length(μ.μg)
eachindex(μ::ConditionalMean) = eachindex(μ.μg)
(μ::ConditionalMean)(x::Number) = map(μ, [x])[1]
(μ::ConditionalMean)(x::AbstractVector) = map(μ, ColsAreObs(reshape(x, length(x), 1)))[1]
function _map(μ::ConditionalMean, Xg::AV)
    return map(μ.μg, Xg) .+ pairwise(μ.kfg, :, Xg)' * μ.c.α
end

"""
    ConditionalKernel <: Kernel

Computes the (co)variance of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalKernel <: Kernel
    c::CondCache
    kfg::CrossKernel
    kgg::Kernel
end
(k::ConditionalKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
(k::ConditionalKernel)(x::AV, x′::AV) =
    map(k, ColsAreObs(reshape(x, length(x), 1)), ColsAreObs(reshape(x′, length(x′), 1)))[1]
length(k::ConditionalKernel) = length(k.kgg)
eachindex(k::ConditionalKernel) = eachindex(k.kgg)

function _map(k::ConditionalKernel, X::AV)
    σgg = map(k.kgg, X)
    Σfg_X = pairwise(k.kfg, :, X)
    σ′gg = diag_Xᵀ_invA_X(k.c.Σff, Σfg_X)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end
function _map(k::ConditionalKernel, X::AV, X′::AV)
    σgg = map(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, :, X)
    Σfg_X′ = pairwise(k.kfg, :, X′)
    σ′gg = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfg_X′)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end

function _pairwise(k::ConditionalKernel, X::AV)
    Σgg = AbstractMatrix(pairwise(k.kgg, X))
    Σfg_X = pairwise(k.kfg, :, X)
    Σ′gg = AbstractMatrix(Xt_invA_X(k.c.Σff, Σfg_X))
    return (Σgg .- Σ′gg) .* .!(Σgg .≈ Σ′gg)
end
function _pairwise(k::ConditionalKernel, X::AV, X′::AV)
    Σgg = pairwise(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, :, X)
    Σfg_X′ = pairwise(k.kfg, :, X′)
    Σ′gg = Xt_invA_Y(Σfg_X, k.c.Σff, Σfg_X′)
    return (Σgg .- Σ′gg) .* .!(Σgg .≈ Σ′gg)
end

"""
    ConditionalCrossKernel <: CrossKernel

Computes the covariance between `g` and `h` conditioned on observations of a process `f`.
"""
struct ConditionalCrossKernel <: CrossKernel
    c::CondCache
    kfg::CrossKernel
    kfh::CrossKernel
    kgh::CrossKernel
end
(k::ConditionalCrossKernel)(x::Number, x′::Number) = map(k, [x], [x′])[1]
(k::ConditionalCrossKernel)(x::AV, x′::AV) =
    map(k, ColsAreObs(reshape(x, length(x), 1)), ColsAreObs(reshape(x′, length(x′), 1)))[1]
size(k::ConditionalCrossKernel, N::Int) = size(k.kgh, N)
eachindex(k::ConditionalCrossKernel, N::Int) = eachindex(k.kgh, N)

function _map(k::ConditionalCrossKernel, X::AV, X′::AV)
    σgh = map(k.kgh, X, X′)
    Σfg_X = pairwise(k.kfg, :, X)
    Σfh_X′ = pairwise(k.kfh, :, X′)
    σ′gh = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfh_X′)
    return (σgh .- σ′gh) .* .!(σgh .≈ σ′gh)
end

function _pairwise(k::ConditionalCrossKernel, Xg::AV, Xh::AV)
    Σgh = pairwise(k.kgh, Xg, Xh)
    Σfg_Xg = pairwise(k.kfg, :, Xg)
    Σfh_Xh = pairwise(k.kfh, :, Xh)
    Σ′gh = Xt_invA_Y(Σfg_Xg, k.c.Σff, Σfh_Xh)
    return (Σgh .- Σ′gh) .* .!(Σgh .≈ Σ′gh)
end
