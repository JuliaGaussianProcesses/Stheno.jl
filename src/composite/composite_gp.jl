"""
    CompositeGP{Targs} <: AbstractGP

A GP derived from other GPs via an affine transformation. Specification given by `args`.
You should generally _not_ construct this object manually.
"""
struct CompositeGP{Targs} <: AbstractGP
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
cov_diag(f::CompositeGP, x::AbstractVector) = cov_diag(f.args, x)

cov(f::CompositeGP, x::AbstractVector, x′::AbstractVector) = cov(f.args, x, x′)
cov_diag(f::CompositeGP, x::AbstractVector, x′::AbstractVector) = cov_diag(f.args, x, x′)

function cov(
    f::Union{WrappedGP, CompositeGP, PPGP},
    f′::Union{WrappedGP, CompositeGP, PPGP},
    x::AbstractVector,
    x′::AbstractVector,
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

function cov_diag(
    f::Union{WrappedGP, CompositeGP, PPGP},
    f′::Union{WrappedGP, CompositeGP, PPGP},
    x::AbstractVector,
    x′::AbstractVector,
)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov_diag(f.args, x, x′)
    elseif f isa WrappedGP && f.n > f′.n || f′ isa WrappedGP && f′.n > f.n
        return zeros(length(x))
    elseif f.n >= f′.n
        return cov_diag(f.args, f′, x, x′)
    else
        return cov_diag(f, f′.args, x, x′)
    end
end

mean_and_cov(f::CompositeGP, x::AbstractVector) = (mean(f, x), cov(f, x))

mean_and_cov_diag(f::CompositeGP, x::AbstractVector) = (mean(f, x), cov_diag(f, x))

# Ensure that cross-covariance computations are handled by this package for CompositeGPs.
function cov(
    fx::FiniteGP{<:Union{WrappedGP, CompositeGP}},
    gx::FiniteGP{<:Union{WrappedGP, CompositeGP}},
)
    return cov(fx.f, gx.f, fx.x, gx.x)
end
