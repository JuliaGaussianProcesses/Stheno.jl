"""
    DerivedGP{Targs} <: AbstractGP

A GP derived from other GPs via an affine transformation. Specification given by `args`.
You should generally _not_ construct this object manually.
"""
struct DerivedGP{Targs} <: SthenoAbstractGP
    args::Targs
    n::Int
    gpc::GPC
    function DerivedGP{Targs}(args::Targs, gpc::GPC) where {Targs}
        gp = new{Targs}(args, next_index(gpc), gpc)
        gpc.n += 1
        return gp
    end
end
DerivedGP(args::Targs, gpc::GPC) where {Targs} = DerivedGP{Targs}(args, gpc)

@opt_out rrule(::typeof(mean), ::DerivedGP, ::AbstractVector)
@opt_out rrule(::typeof(cov), ::DerivedGP, ::AbstractVector)
@opt_out rrule(::typeof(var), ::DerivedGP, ::AbstractVector)

AbstractGPs.mean(f::DerivedGP, x::AbstractVector) = mean(f.args, x)

AbstractGPs.cov(f::DerivedGP, x::AbstractVector) = cov(f.args, x)
AbstractGPs.var(f::DerivedGP, x::AbstractVector) = var(f.args, x)

AbstractGPs.cov(f::DerivedGP, x::AbstractVector, x′::AbstractVector) = cov(f.args, x, x′)
AbstractGPs.var(f::DerivedGP, x::AbstractVector, x′::AbstractVector) = var(f.args, x, x′)

function AbstractGPs.cov(
    f::SthenoAbstractGP, f′::SthenoAbstractGP, x::AbstractVector, x′::AbstractVector,
)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov(f.args, x, x′)
    elseif f isa AtomicGP && f.n > f′.n || f′ isa AtomicGP && f′.n > f.n
        return zeros(length(x), length(x′))
    elseif f.n >= f′.n
        return cov(f.args, f′, x, x′)
    else
        return cov(f, f′.args, x, x′)
    end
end

function AbstractGPs.var(
    f::SthenoAbstractGP, f′::SthenoAbstractGP, x::AbstractVector, x′::AbstractVector,
)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return var(f.args, x, x′)
    elseif f isa AtomicGP && f.n > f′.n || f′ isa AtomicGP && f′.n > f.n
        return zeros(length(x))
    elseif f.n >= f′.n
        return var(f.args, f′, x, x′)
    else
        return var(f, f′.args, x, x′)
    end
end
