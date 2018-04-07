export ConditionalMean, ConditionalKernel, ConditionalCrossKernel

"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
"""
struct CondCache
    kff::Kernel
    μf::MeanFunction
    f::AV{<:Real}
    α::AV{<:Real}
    function CondCache(kff::Kernel, μf::MeanFunction, f::AV{<:Real})
        @assert isfinite(kff)
        @assert isfinite(μf)
        @assert length(μf) == length(f)
        @assert size(kff, 1) == size(μf)
        return new(kff, μf, f, kff \ (f - μf))
    end
end

"""
    ConditionalMean <: Function

The function defining the mean of the process `g | (f(X) ← f)`. `Xf` are observation
locations, `f` are the observed values, `kff` is the covariance function of `f`, and `kgf`
is the cross-covariance between `f` and `g`. `μf` and `μg` are the prior mean for `f` and
`g` resp.
"""
struct ConditionalMean <: Function
    c::CondCache
    μg::MeanFunction
    kgf::CrossKernel
    function ConditionalMean(c::CondCache, μg::MeanFunction, kgf::CrossKernel)
        if isfinite(μg)
            @assert length(μg) == size(kgf, 1)
        else
            return ConditionalMean(c, μg, kgf)
        end
    end
end
isfinite(μ::ConditionalMean) = isfinite(μ.μg)
(μ::ConditionalMean)(x) = mean(μ, reshape(x, length(x), 1))
mean(μ::ConditionalMean, Xg::AM) = xcov(μ.kgf, Xg, μ.kff.X) * μ.c.α + mean(μ.μg, Xg)
mean(μ::ConditionalMean) = xcov(μ.kgf) * μ.c.α + mean(μ.μg)

"""
    ConditionalKernel <: Kernel

A `ConditionalKernel` is used to compute the covariance / xcov betweeen points in a process
`g` once it has been conditioned on observations from some other process `f`.
"""
struct ConditionalKernel <: Kernel
    kff::FiniteKernel
    kfg::CrossKernel
    kgg::Kernel
end
isstationary(k::ConditionalKernel) = false
isfinite(k::ConditionalKernel) = isfinite(k.kgg)
function cov(k::ConditionalKernel, X::AM)
    Σff, Σgg, Σfg = cov(k.kff), cov(k.kgg, X), xcov(k.kfg, args(k.kff), X)
    return Σgg - Xt_invA_X(Σff, Σfg)
end
function xcov(k::ConditionalKernel, X::AM, X′::AM)
    Σff, Σgg′ = cov(k.kff), xcov(k.kgg, X, X′)
    Σfg, Σfg′ = xcov(k.kfg, k.kff.X, X), xcov(k.kfg, k.kff.X, X′)
    return Σgg′ - Xt_invA_Y(Σfg, Σff, Σfg′)
end

"""
    ConditionalCrossKernel <: CrossKernel

A conditional cross kernel is used to compute the xcov between two processes `g` and `h`,
conditioned on observations of a third process `f`
"""
struct ConditionalCrossKernel <: CrossKernel
    kff::FiniteKernel
    kfg::CrossKernel
    kfh::CrossKernel
    kgh::CrossKernel
end
isstationary(k::ConditionalCrossKernel) = false
isfinite(k::ConditionalCrossKernel) = isfinite(k.kgh)
xcov(k::ConditionalCrossKernel, X::AM) = xcov(k, X, X)
function xcov(k::ConditionalCrossKernel, Xg::AM, Xh::AM)
    Σff, Σgh = cov(k.kff), xcov(k.kgh, Xg, Xh)
    Σfg, Σfh = xcov(k.kfg, k.kff.X, Xg), xcov(k.kfh, k.kff.X, Xh) 
    return Σgh - Xt_invA_Y(Σfg, Σff, Σfh)
end

# The kernel function is defined identically for both types of conditionals.
for Foo in [:ConditionalKernel, :ConditionalCrossKernel]
    @eval function (k::$Foo)(x, x′)
        X, X′ = reshape(x, 1, length(x)), reshape(x′, 1, length(x′))
        return xcov(k, X, X′)
    end
end
