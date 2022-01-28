"""
    AtomicGP{Tgp<:AbstractGP} <: SthenoAbstractGP

A thin wrapper around an AbstractGP that does some book-keeping.

```julia
f = atomic(GP(SEKernel()), GPC())
```
builds a `AtomicGP` that `Stheno` knows how to work with.
"""
struct AtomicGP{Tgp<:AbstractGP} <: SthenoAbstractGP
    gp::Tgp
    n::Int
    gpc::GPC
    function AtomicGP{Tgp}(gp::Tgp, gpc::GPC) where {Tgp<:AbstractGP}
        wgp = new{Tgp}(gp, next_index(gpc), gpc)
        gpc.n += 1
        return wgp
    end
end

atomic(gp::Tgp, gpc::GPC) where {Tgp<:AbstractGP} = AtomicGP{Tgp}(gp, gpc)

mean(f::AtomicGP, x::AbstractVector) = mean(f.gp, x)

cov(f::AtomicGP, x::AbstractVector) = cov(f.gp, x)
var(f::AtomicGP, x::AbstractVector) = var(f.gp, x)

cov(f::AtomicGP, x::AbstractVector, x′::AbstractVector) = cov(f.gp, x, x′)
var(f::AtomicGP, x::AbstractVector, x′::AbstractVector) = var(f.gp, x, x′)

function cov(f::AtomicGP, f′::AtomicGP, x::AbstractVector, x′::AbstractVector)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function var(f::AtomicGP, f′::AtomicGP, x::AbstractVector, x′::AbstractVector)
    return f === f′ ? var(f, x, x′) : zeros(length(x))
end
