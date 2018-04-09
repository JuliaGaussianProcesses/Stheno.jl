export ConditionalMean, ConditionalKernel, ConditionalCrossKernel

"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache
    Σff::AbstractPDMat
    α::AV{<:Real}
    function CondCache(kff::Kernel, μf::MeanFunction, f::AV{<:Real})
        Σff = cov(kff)
        return new(Σff, Σff \ (f - mean(μf)))
    end
end

"""
    ConditionalMean <: Function

The function defining the mean of the process `g | (f(X) ← f)`. `Xf` are observation
locations, `f` are the observed values, `kff` is the covariance function of `f`, and `kfg`
is the cross-covariance between `f` and `g`. `μf` and `μg` are the prior mean for `f` and
`g` resp.
"""
struct ConditionalMean <: Function
    c::CondCache
    μg::MeanFunction
    kfg::CrossKernel
    function ConditionalMean(c::CondCache, μg::MeanFunction, kfg::CrossKernel)
        isfinite(μg) && @assert length(μg) == size(kfg, 1)
        return ConditionalMean(c, μg, kfg)
    end
end
isfinite(μ::ConditionalMean) = isfinite(μ.μg)
(μ::ConditionalMean)(x) = mean(μ, reshape(x, 1, length(x)))
mean(μ::ConditionalMean, Xg::AM) = mean(μ.μg, Xg) + xcov(μ.kfg, Xg)' * μ.c.α
mean(μ::ConditionalMean) = mean(μ.μg) + xcov(μ.kfg)' * μ.c.α

"""
    ConditionalKernel <: Kernel

A `ConditionalKernel` is used to compute the covariance / xcov betweeen points in a process
`g` once it has been conditioned on observations from some other process `f`. `f` must be a
finite dimensional GP.
"""
struct ConditionalKernel <: Kernel
    c::CondCache
    kfg::CrossKernel
    kgg::Kernel
end
isstationary(k::ConditionalKernel) = false
isfinite(k::ConditionalKernel) = isfinite(k.kgg)
cov(k::ConditionalKernel) = cov(k.kgg) - Xt_invA_X(k.c.Σff, xcov(k.kfg))
cov(k::ConditionalKernel, X::AM) = cov(k.kgg, X) - Xt_invA_X(k.c.Σff, xcov(k.kfg, X))
xcov(k::ConditionalKernel, X::AM, X′::AM) =
    xcov(k.kgg, X, X′) - Xt_invA_Y(xcov(k.kfg, X), k.c.Σff, xcov(k.kfg, X′))

"""
    ConditionalCrossKernel <: CrossKernel

A conditional cross kernel is used to compute the xcov between two processes `g` and `h`,
conditioned on observations of a third process `f`
"""
struct ConditionalCrossKernel <: CrossKernel
    c::CondCache
    kfg::CrossKernel
    kfh::CrossKernel
    kgh::CrossKernel
end
isstationary(k::ConditionalCrossKernel) = false
isfinite(k::ConditionalCrossKernel) = isfinite(k.kgh)
xcov(k::ConditionalCrossKernel, X::AM) = xcov(k, X, X)
xcov(k::ConditionalCrossKernel, Xg::AM, Xh::AM) =
    xcov(k.kgh, Xg, Xh) - Xt_invA_Y(xcov(k.kfg, Xg), k.c.Σff, xcov(k.kfh, Xh))

# The kernel function is defined identically for both types of conditionals.
for Foo in [:ConditionalKernel, :ConditionalCrossKernel]
    @eval function (k::$Foo)(x, x′)
        X, X′ = reshape(x, 1, length(x)), reshape(x′, 1, length(x′))
        return xcov(k, X, X′)
    end
end
