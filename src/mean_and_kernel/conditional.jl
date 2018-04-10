export ConditionalMean, ConditionalKernel, ConditionalCrossKernel

"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache
    Σff::AbstractPDMat
    α::AV{<:Real}
    X::AM
    function CondCache(kff::Kernel, μf::MeanFunction, X::AM, f::AV{<:Real})
        Σff = cov(kff, X)
        return new(Σff, Σff \ (f - mean(μf, X)), X)
    end
end
show(io::IO, ::CondCache) = show(io, "CondCache")

"""
    ConditionalMean <: MeanFunction

The function defining the mean of the process `g | (f(X) ← f)`. `Xf` are observation
locations, `f` are the observed values, `kff` is the covariance function of `f`, and `kfg`
is the cross-covariance between `f` and `g`. `μf` and `μg` are the prior mean for `f` and
`g` resp.
"""
struct ConditionalMean <: MeanFunction
    c::CondCache
    μg::MeanFunction
    kfg::CrossKernel
end
length(μ::ConditionalMean) = length(μ.μg)
(μ::ConditionalMean)(x) = mean(μ, reshape(x, 1, length(x)))
mean(μ::ConditionalMean, Xg::AM) = mean(μ.μg, Xg) + xcov(μ.kfg, μ.c.X, Xg)' * μ.c.α

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
cov(k::ConditionalKernel, X::AM) = cov(k.kgg, X) - Xt_invA_X(k.c.Σff, xcov(k.kfg, k.c.X, X))
xcov(k::ConditionalKernel, X::AM, X′::AM) =
    xcov(k.kgg, X, X′) - Xt_invA_Y(xcov(k.kfg, k.c.X, X), k.c.Σff, xcov(k.kfg, k.c.X, X′))

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
xcov(k::ConditionalCrossKernel, X::AM) = xcov(k, X, X)
xcov(k::ConditionalCrossKernel, Xg::AM, Xh::AM) =
    xcov(k.kgh, Xg, Xh) - Xt_invA_Y(xcov(k.kfg, k.c.X, Xg), k.c.Σff, xcov(k.kfh, k.c.X, Xh))

# The kernel function is defined identically for both types of conditionals.
for Foo in [:ConditionalKernel, :ConditionalCrossKernel]
    @eval function (k::$Foo)(x, x′)
        X, X′ = reshape(x, 1, length(x)), reshape(x′, 1, length(x′))
        return xcov(k, X, X′)
    end
end
