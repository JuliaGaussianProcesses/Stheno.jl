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
    CondCache(mf::AV{<:Real}, Σff::AM, f::AV{<:Real}) = new(Σff, Σff \ (f - mf))
end
function CondCache(kff::Kernel, μf::MeanFunction, X::ADS, f::AV{<:Real})
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
(μ::ConditionalMean)(x::Number) = map(μ, DataSet([x]))[1]
(μ::ConditionalMean)(x::AbstractVector) = map(μ, DataSet(reshape(x, length(x), 1)))[1]
function map(μ::ConditionalMean, Xg::DataSet)
    return map(μ.μg, Xg) + pairwise(μ.kfg, ADS(eachindex(μ.kfg, 1)), Xg)' * μ.c.α
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
(k::ConditionalKernel)(x::Number, x′::Number) = map(k, DataSet([x]), DataSet([x′]))[1]
(k::ConditionalKernel)(x::AV, x′::AV) =
    map(k, DataSet(reshape(x, length(x), 1)), DataSet(reshape(x′, length(x′), 1)))[1]
size(k::ConditionalKernel, N::Int) = size(k.kgg, N)
eachindex(k::ConditionalKernel) = eachindex(k.kgg)

function map(k::ConditionalKernel, X::DataSet)
    σgg = map(k.kgg, X)
    Σfg_X = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X)
    σ′gg = diag_Xᵀ_invA_X(k.c.Σff, Σfg_X)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end
function map(k::ConditionalKernel, X::DataSet, X′::DataSet)
    σgg = map(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X)
    Σfg_X′ = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X′)
    σ′gg = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfg_X′)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end

function pairwise(k::ConditionalKernel, X::DataSet)
    Σgg = AbstractMatrix(pairwise(k.kgg, X))
    Σfg_X = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X)
    Σ′gg = AbstractMatrix(Xt_invA_X(k.c.Σff, Σfg_X))
    return LazyPDMat((Σgg .- Σ′gg) .* .!(Σgg .≈ Σ′gg))
end
function pairwise(k::ConditionalKernel, X::DataSet, X′::DataSet)
    Σgg = pairwise(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X)
    Σfg_X′ = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X′)
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
(k::ConditionalCrossKernel)(x::Number, x′::Number) = map(k, DataSet([x]), DataSet([x′]))[1]
(k::ConditionalCrossKernel)(x::AV, x′::AV) =
    map(k, DataSet(reshape(x, length(x), 1)), DataSet(reshape(x′, length(x′), 1)))[1]
size(k::ConditionalCrossKernel, N::Int) = size(k.kgh, N)
eachindex(k::ConditionalCrossKernel, N::Int) = eachindex(k.kgh, N)

function map(k::ConditionalCrossKernel, X::DataSet, X′::DataSet)
    σgh = map(k.kgh, X, X′)
    Σfg_X = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), X)
    Σfh_X′ = pairwise(k.kfh, ADS(eachindex(k.kfh, 1)), X′)
    σ′gh = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfh_X′)
    return (σgh .- σ′gh) .* .!(σgh .≈ σ′gh)
end

function pairwise(k::ConditionalCrossKernel, Xg::DataSet, Xh::DataSet)
    Σgh = pairwise(k.kgh, Xg, Xh)
    Σfg_Xg = pairwise(k.kfg, ADS(eachindex(k.kfg, 1)), Xg)
    Σfh_Xh = pairwise(k.kfh, ADS(eachindex(k.kfh, 1)), Xh)
    Σ′gh = Xt_invA_Y(Σfg_Xg, k.c.Σff, Σfh_Xh)
    return (Σgh .- Σ′gh) .* .!(Σgh .≈ Σ′gh)
end
