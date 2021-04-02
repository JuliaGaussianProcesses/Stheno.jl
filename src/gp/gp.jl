"""
    WrappedGP{Tgp<:AbstractGP} <: SthenoAbstractGP

A thin wrapper around an AbstractGP that does some book-keeping.

```julia
f = wrap(GP(SEKernel()), GPC())
```
builds a `WrappedGP` that `Stheno` knows how to work with.
"""
struct WrappedGP{Tgp<:AbstractGP} <: SthenoAbstractGP
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

mean(f::WrappedGP, x::AbstractVector) = mean(f.gp, x)

cov(f::WrappedGP, x::AbstractVector) = cov(f.gp, x)
var(f::WrappedGP, x::AbstractVector) = var(f.gp, x)

cov(f::WrappedGP, x::AbstractVector, x′::AbstractVector) = cov(f.gp, x, x′)
var(f::WrappedGP, x::AbstractVector, x′::AbstractVector) = var(f.gp, x, x′)

function cov(f::WrappedGP, f′::WrappedGP, x::AbstractVector, x′::AbstractVector)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function var(f::WrappedGP, f′::WrappedGP, x::AbstractVector, x′::AbstractVector)
    return f === f′ ? var(f, x, x′) : zeros(length(x))
end
