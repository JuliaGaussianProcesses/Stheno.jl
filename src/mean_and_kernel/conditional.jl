export ConditionalMean, ConditionalKernel, ConditionalCrossKernel

"""
    ConditionalMean <: Function

The function defining the mean of the process `g | (f(X) ← f)`. `Xf` are observation
locations, `f` are the observed values, `kff` is the covariance function of `f`, and `kgf`
is the cross-covariance between `f` and `g`. `μf` and `μg` are the prior mean for `f` and
`g` resp.
"""
struct ConditionalMean <: Function
    kff::FiniteKernel
    μf::Function
    f::AbstractVector{<:Real}
    μg::Function
    kgf::CrossKernel
end
(μ::ConditionalMean)(x) = mean(μ, reshape(x, length(x), 1))
function mean(μ::ConditionalMean, Xg::AM)
    U = chol(cov(μ.kff))
    α = U \ (U' \ (μ.f - μ.μf(X)))
    return μ.kgf(Xg, μ.kff.X) * α + μ.μg(Xg)
end

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
