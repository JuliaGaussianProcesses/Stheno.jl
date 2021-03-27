"""
    WrappedGP{Tgp<:AbstractGP} <: AbstractGP

A thin wrapper around a GP that does some book-keeping.
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

mean_vector(f::WrappedGP, x::AV) = mean_vector(f.gp, x)

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
