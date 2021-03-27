"""
    WrappedGP{Tgp<:AbstractGP} <: AbstractGP

A thin wrapper around an AbstractGP that does some book-keeping.

```julia
f = wrap(GP(SEKernel()), GPC())
```
builds a `WrappedGP` that `Stheno` knows how to work with.
"""
struct WrappedGP{Tgp<:AbstractGP} <: AbstractGP
    gp::Tgp
    n::Int
    gpc::GPC
    function WrappedGP{Tgp}(gp::Tgp, gpc::GPC) where {Tgp<:GP}
        wgp = new{Tgp}(gp, next_index(gpc), gpc)
        gpc.n += 1
        return wgp
    end
end

wrap(gp::Tgp, gpc::GPC) where {Tgp<:GP} = WrappedGP{Tgp}(gp, gpc)

mean(f::WrappedGP, x::AV) = mean(f.gp, x)

cov(f::WrappedGP, x::AV) = cov(f.gp, x)
cov_diag(f::WrappedGP, x::AV) = cov_diag(f.gp, x)

cov(f::WrappedGP, x::AV, x′::AV) = cov(f.gp, x, x′)
cov_diag(f::WrappedGP, x::AV, x′::AV) = cov_diag(f.gp, x, x′)

function cov(f::WrappedGP, f′::WrappedGP, x::AV, x′::AV)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function cov_diag(f::WrappedGP, f′::WrappedGP, x::AV, x′::AV)
    return f === f′ ? cov_diag(f, x, x′) : zeros(length(x))
end

# Ensure that cross-covariance computations are handled by this package for the WrappedGPs.
cov(fx::FiniteGP{<:WrappedGP}, gx::FiniteGP{<:WrappedGP}) = cov(fx.f, gx.f, fx.x, gx.x)
