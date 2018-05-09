export ConditionalMean, ConditionalKernel, ConditionalCrossKernel

"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache
    Σff::LazyPDMat
    α::AV{<:Real}
    X::AVM
    function CondCache(kff::Kernel, μf::MeanFunction, X::AVM, f::AV{<:Real})
        μfX, Σff = unary_obswise(μf, X), LazyPDMat(pairwise(kff, X))
        δ = (f .- μfX) .* .!(f .≈ μfX)
        return new(Σff, Σff \ δ, X)
    end
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
(μ::ConditionalMean)(x::Number) = unary_obswise(μ, [x])[1]
(μ::ConditionalMean)(x::AbstractVector) = unary_obswise(μ, reshape(x, length(x), 1))[1]
unary_obswise(μ::ConditionalMean, Xg::AVM) =
    unary_obswise(μ.μg, Xg) + pairwise(μ.kfg, μ.c.X, Xg)' * μ.c.α

"""
    ConditionalKernel <: Kernel

Computes the (co)variance of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalKernel <: Kernel
    c::CondCache
    kfg::CrossKernel
    kgg::Kernel
end
(k::ConditionalKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
(k::ConditionalKernel)(x::AV, x′::AV) =
    binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
size(k::ConditionalKernel, N::Int) = size(k.kgg, N)

function binary_obswise(k::ConditionalKernel, X::AVM)
    σgg = binary_obswise(k.kgg, X)
    Σfg_X = pairwise(k.kfg, k.c.X, X)
    σ′gg = diag_Xᵀ_invA_X(k.c.Σff, Σfg_X)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end
function binary_obswise(k::ConditionalKernel, X::AVM, X′::AVM)
    σgg = binary_obswise(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, k.c.X, X)
    Σfg_X′ = pairwise(k.kfg, k.c.X, X′)
    σ′gg = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfg_X′)
    return (σgg .- σ′gg) .* .!(σgg .≈ σ′gg)
end

function pairwise(k::ConditionalKernel, X::AVM)
    Σgg = pairwise(k.kgg, X)
    Σfg_X = pairwise(k.kfg, k.c.X, X)
    Σ′gg = Xt_invA_X(k.c.Σff, Σfg_X)
    return (Σgg .- Σ′gg) .* .!(Σgg .≈ Σ′gg)
end
function pairwise(k::ConditionalKernel, X::AVM, X′::AVM)
    Σgg = pairwise(k.kgg, X, X′)
    Σfg_X = pairwise(k.kfg, k.c.X, X)
    Σfg_X′ = pairwise(k.kfg, k.c.X, X′)
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
(k::ConditionalCrossKernel)(x::Number, x′::Number) = binary_obswise(k, [x], [x′])[1]
(k::ConditionalCrossKernel)(x::AV, x′::AV) =
    binary_obswise(k, reshape(x, length(x), 1), reshape(x′, length(x′), 1))[1]
size(k::ConditionalCrossKernel, N::Int) = size(k.kgh, N)

function binary_obswise(k::ConditionalCrossKernel, X::AVM, X′::AVM)
    σgh = binary_obswise(k.kgh, X, X′)
    Σfg_X = pairwise(k.kfg, k.c.X, X)
    Σfh_X′ = pairwise(k.kfh, k.c.X, X′)
    σ′gh = diag_Xᵀ_invA_Y(Σfg_X, k.c.Σff, Σfh_X′)
    return (σgh .- σ′gh) .* .!(σgh .≈ σ′gh)
end

function pairwise(k::ConditionalCrossKernel, Xg::AVM, Xh::AVM)
    Σgh = pairwise(k.kgh, Xg, Xh)
    Σfg_Xg = pairwise(k.kfg, k.c.X, Xg)
    Σfh_Xh = pairwise(k.kfh, k.c.X, Xh)
    Σ′gh = Xt_invA_Y(Σfg_Xg, k.c.Σff, Σfh_Xh)
    return (Σgh .- Σ′gh) .* .!(Σgh .≈ Σ′gh)
end
