"""
    CondCache

Cache for use by `CondMean`s, `CondKernel`s and `CondCrossKernel`s.
Avoids recomputing the covariance `Σff` and the Kriging vector `α`.
"""
struct CondCache{TC<:Cholesky, Tα<:AbstractVector{<:Real}, Tx<:AbstractVector}
    C::TC
    α::Tα
    x::Tx
end
function CondCache(
    kff::Kernel,
    μf::MeanFunction,
    x::AV,
    y::AV{<:Real},
    σ²::Union{Real, AV{<:Real}},
)
    C = cholesky(pw(kff, x) + _get_mat(σ²))
    return CondCache(C, C \ (y - map(μf, x)), x)
end
_get_mat(σ²::Real) = σ² * I
_get_mat(σ²::AV{<:Real}) = Diagonal(σ²)

"""
    CondMean <: MeanFunction

Computes the mean of a GP `g` conditioned on observations of another GP `f`.
"""
struct CondMean{Tc<:CondCache, Tμg<:MeanFunction, Tkfg<:CrossKernel} <: MeanFunction
    c::Tc
    μg::Tμg
    kfg::Tkfg
end
_map(μ::CondMean, xg::AV) = bcd(+, _map(μ.μg, xg), pw(μ.kfg, μ.c.x, xg)' * μ.c.α)


"""
    CondKernel <: Kernel

Computes the (co)variance of a GP `g` conditioned on observations of another GP `f`.
"""
struct CondKernel{Tc<:CondCache, Tkfg<:CrossKernel, Tkgg<:Kernel} <: Kernel
    c::Tc
    kfg::Tkfg
    kgg::Tkgg
end

# Binary methods.
function _map(k::CondKernel, x::AV, x′::AV)
    σgg = map(k.kgg, x, x′)
    σ′gg = diag_Xt_invA_Y(pw(k.kfg, k.c.x, x), k.c.C, pw(k.kfg, k.c.x, x′))
    return σgg - σ′gg
end
function _pw(k::CondKernel, x::AV, x′::AV)
    Σgg = pw(k.kgg, x, x′)
    Σ′gg = Xt_invA_Y(pw(k.kfg, k.c.x, x), k.c.C, pw(k.kfg, k.c.x, x′))
    return Σgg - Σ′gg
end

# Unary methods.
function _map(k::CondKernel, x::AV)
    σgg = map(k.kgg, x)
    σ′gg = diag_Xt_invA_X(k.c.C, pw(k.kfg, k.c.x, x))
    return σgg - σ′gg
end
function _pw(k::CondKernel, x::AV)
    Σgg = pw(k.kgg, x)
    Σ′gg = Xt_invA_X(k.c.C, pw(k.kfg, k.c.x, x))
    return Σgg - Σ′gg
end


"""
    CondCrossKernel <: CrossKernel

Computes the covariance between `g` and `h` conditioned on observations of a process `f`.
"""
struct CondCrossKernel{
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

function _map(k::CondCrossKernel, x::AV, x′::AV)
    σgh = map(k.kgh, x, x′)
    σ′gh = diag_Xt_invA_Y(pw(k.kfg, k.c.x, x), k.c.C, pw(k.kfh, k.c.x, x′))
    return σgh - σ′gh
end
function _pw(k::CondCrossKernel, xg::AV, xh::AV)
    Σgh = pw(k.kgh, xg, xh)
    Σ′gh = Xt_invA_Y(pw(k.kfg, k.c.x, xg), k.c.C, pw(k.kfh, k.c.x, xh))
    return Σgh - Σ′gh
end
