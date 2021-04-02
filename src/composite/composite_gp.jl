"""
    CompositeGP{Targs} <: AbstractGP

A GP derived from other GPs via an affine transformation. Specification given by `args`.
You should generally _not_ construct this object manually.
"""
struct CompositeGP{Targs} <: SthenoAbstractGP
    args::Targs
    n::Int
    gpc::GPC
    function CompositeGP{Targs}(args::Targs, gpc::GPC) where {Targs}
        gp = new{Targs}(args, next_index(gpc), gpc)
        gpc.n += 1
        return gp
    end
end
CompositeGP(args::Targs, gpc::GPC) where {Targs} = CompositeGP{Targs}(args, gpc)

mean(f::CompositeGP, x::AbstractVector) = mean(f.args, x)

cov(f::CompositeGP, x::AbstractVector) = cov(f.args, x)
var(f::CompositeGP, x::AbstractVector) = var(f.args, x)

cov(f::CompositeGP, x::AbstractVector, x′::AbstractVector) = cov(f.args, x, x′)
var(f::CompositeGP, x::AbstractVector, x′::AbstractVector) = var(f.args, x, x′)

function cov(
    f::SthenoAbstractGP, f′::SthenoAbstractGP, x::AbstractVector, x′::AbstractVector,
)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov(f.args, x, x′)
    elseif f isa WrappedGP && f.n > f′.n || f′ isa WrappedGP && f′.n > f.n
        return zeros(length(x), length(x′))
    elseif f.n >= f′.n
        return cov(f.args, f′, x, x′)
    else
        return cov(f, f′.args, x, x′)
    end
end

function var(
    f::SthenoAbstractGP, f′::SthenoAbstractGP, x::AbstractVector, x′::AbstractVector,
)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return var(f.args, x, x′)
    elseif f isa WrappedGP && f.n > f′.n || f′ isa WrappedGP && f′.n > f.n
        return zeros(length(x))
    elseif f.n >= f′.n
        return var(f.args, f′, x, x′)
    else
        return var(f, f′.args, x, x′)
    end
end
