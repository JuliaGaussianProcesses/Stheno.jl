"""
    CompositeGP{Targs}

A GP derived from other GPs via an affine transformation. Specification given by `args`.
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

mean_vector(f::CompositeGP, x::AV) = mean_vector(f.args, x)

cov(f::CompositeGP, x::AV) = cov(f.args, x)
cov_diag(f::CompositeGP, x::AV) = cov_diag(f.args, x)

cov(f::CompositeGP, x::AV, x′::AV) = cov(f.args, x, x′)
cov_diag(f::CompositeGP, x::AV, x′::AV) = cov_diag(f.args, x, x′)

function cov(f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov(f.args, x, x′)
    elseif f isa GP && f.n > f′.n || f′ isa GP && f′.n > f.n
        return zeros(length(x), length(x′))
    elseif f.n >= f′.n
        return cov(f.args, f′, x, x′)
    else
        return cov(f, f′.args, x, x′)
    end
end
function cov_diag(f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov_diag(f.args, x, x′)
    elseif f isa GP && f.n > f′.n || f′ isa GP && f′.n > f.n
        return zeros(length(x))
    elseif f.n >= f′.n
        return cov_diag(f.args, f′, x, x′)
    else
        return cov_diag(f, f′.args, x, x′)
    end
end
