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
        Σff = cov(kff, X)
        return new(Σff, Σff \ (f - mean(μf, X)), X)
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
mean(μ::ConditionalMean, Xg::AVM) = mean(μ.μg, Xg) + xcov(μ.kfg, μ.c.X, Xg)' * μ.c.α

"""
    ConditionalKernel <: Kernel

Computes the (co)variance of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalKernel <: Kernel
    c::CondCache
    kfg::CrossKernel
    kgg::Kernel
end
size(k::ConditionalKernel, N::Int) = size(k.kgg, N)
function cov(k::ConditionalKernel, X::AVM)
    Σfg, Σgg = xcov(k.kfg, k.c.X, X), cov(k.kgg, X)
    return Σgg - Xt_invA_X(k.c.Σff, Σfg)
end
function xcov(k::ConditionalKernel, X::AVM, X′::AVM)
    Σfg_X, Σfg_X′, Σfg_XX′ = xcov(k.kfg, k.c.X, X), xcov(k.kfg, k.c.X, X′), xcov(k.kgg, X, X′)
    return Σfg_XX′ - Xt_invA_Y(Σfg_X, k.c.Σff, Σfg_X′)
end
function marginal_cov(k::ConditionalKernel, X::AVM)
    Σfg, Σgg = xcov(k.kfg, k.c.X, X), marginal_cov(k.kgg, X)
    return Σgg - vec(sum(abs2, Σfg' / chol(k.c.Σff), 2))
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
size(k::ConditionalCrossKernel, N::Int) = size(k.kgh, N)
function xcov(k::ConditionalCrossKernel, Xg::AVM, Xh::AVM)
    Σfg, Σfh, Σgh = xcov(k.kfg, k.c.X, Xg), xcov(k.kfh, k.c.X, Xh), xcov(k.kgh, Xg, Xh)
    return Σgh - Xt_invA_Y(Σfg, k.c.Σff, Σfh)
end
