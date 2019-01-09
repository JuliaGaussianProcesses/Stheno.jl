"""
    CondCache

Cache for use by `ConditionalMean`s, `ConditionalKernel`s and `ConditionalCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache{TC<:Cholesky, Tα<:AV{<:Real}}
    C::TC
    α::Tα
end
function CondCache(kff::Kernel, μf::MeanFunction, x::AV, f::AV{<:Real})
    C = cholesky(pw(kff, x))
    return CondCache(C, C \ (f - map(μf, x)))
end

"""
    ConditionalMean <: MeanFunction

Computes the mean of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalMean{Tc<:CondCache, Tμg<:MeanFunction, Tkfg<:CrossKernel} <: MeanFunction
    c::Tc
    μg::Tμg
    kfg::Tkfg
end
(μ::ConditionalMean)(x::Real) = μ.μg(x) + dot(vec(pw(μ.kfg, :, [x])), μ.c.α)
function (μ::ConditionalMean)(x::AV{<:Real})
    return μ.μg(x) + dot(vec(pw(μ.kfg, :, ColsAreObs(reshape(x, :, 1)))), μ.c.α)
end
_map(μ::ConditionalMean, xg::AV) = bcd(+, _map(μ.μg, xg), pw(μ.kfg, :, xg)' * μ.c.α)


"""
    ConditionalKernel <: Kernel

Computes the (co)variance of a GP `g` conditioned on observations of another GP `f`.
"""
struct ConditionalKernel{Tc<:CondCache, Tkfg<:CrossKernel, Tkgg<:Kernel} <: Kernel
    c::Tc
    kfg::Tkfg
    kgg::Tkgg
end

# Binary methods.
(k::ConditionalKernel)(x::Real, x′::Real) = map(k, [x], [x′])[1]
function (k::ConditionalKernel)(x::AV{<:Real}, x′::AV{<:Real})
    return map(k, ColsAreObs(reshape(x, :, 1)), ColsAreObs(reshape(x′, :, 1)))[1]
end
function _map(k::ConditionalKernel, x::AV, x′::AV)
    σgg = map(k.kgg, x, x′)
    σ′gg = diag_Xt_invA_Y(pw(k.kfg, :, x), k.c.C, pw(k.kfg, :, x′))
    return (σgg .- σ′gg)
end
function _pw(k::ConditionalKernel, x::AV, x′::AV)
    Σgg = pw(k.kgg, x, x′)
    Σ′gg = Xt_invA_Y(pw(k.kfg, :, x), k.c.C, pw(k.kfg, :, x′))
    return (Σgg .- Σ′gg)
end

# Unary methods.
(k::ConditionalKernel)(x::Real) = map(k, [x])[1]
(k::ConditionalKernel)(x::AV{<:Real}) = map(k, ColsAreObs(reshape(x, :, 1)))[1]
function _map(k::ConditionalKernel, x::AV)
    σgg = map(k.kgg, x)
    σ′gg = diag_Xt_invA_X(k.c.C, pw(k.kfg, :, x))
    return (σgg .- σ′gg)
end
function _pw(k::ConditionalKernel, x::AV)
    Σgg = pw(k.kgg, x)
    Σ′gg = Xt_invA_X(k.c.C, pw(k.kfg, :, x))
    return (Σgg .- Σ′gg)
end


"""
    ConditionalCrossKernel <: CrossKernel

Computes the covariance between `g` and `h` conditioned on observations of a process `f`.
"""
struct ConditionalCrossKernel{
    Tc<:CondCache,
    Tkfg<:CrossKernel,
    Tkfh<:CrossKernel,
    Tkgh<:CrossKernel,
} <: CrossKernel
    c::Tc
    kfg::Tkfg
    kfh::Tkfh
    kgh::Tkgh
end

(k::ConditionalCrossKernel)(x::Real, x′::Real) = map(k, [x], [x′])[1]
function (k::ConditionalCrossKernel)(x::AV{<:Real}, x′::AV{<:Real})
    return map(k, ColsAreObs(reshape(x, :, 1)), ColsAreObs(reshape(x′, :, 1)))[1]
end
function _map(k::ConditionalCrossKernel, x::AV, x′::AV)
    σgh = map(k.kgh, x, x′)
    σ′gh = diag_Xt_invA_Y(pw(k.kfg, :, x), k.c.C, pw(k.kfh, :, x′))
    return (σgh .- σ′gh)
end
function _pw(k::ConditionalCrossKernel, xg::AV, xh::AV)
    Σgh = pw(k.kgh, xg, xh)
    Σ′gh = Xt_invA_Y(pw(k.kfg, :, xg), k.c.C, pw(k.kfh, :, xh))
    return (Σgh .- Σ′gh)
end
