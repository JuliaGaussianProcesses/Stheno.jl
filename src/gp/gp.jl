export GP

"""
    GP{Tm<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) with known mean `m` and kernel `k`, coordinated by `gpc`.
"""
struct GP{Tm<:MeanFunction, Tk<:Kernel} <: AbstractGP
    m::Tm
    k::Tk
    n::Int
    gpc::GPC
    function GP{Tm, Tk}(m::Tm, k::Tk, gpc::GPC) where {Tm, Tk}
        gp = new{Tm, Tk}(m, k, next_index(gpc), gpc)
        gpc.n += 1
        return gp
    end
end
GP(m::Tm, k::Tk, gpc::GPC) where {Tm<:MeanFunction, Tk<:Kernel} = GP{Tm, Tk}(m, k, gpc)

GP(f, k::Kernel, gpc::GPC) = GP(CustomMean(f), k, gpc)
GP(m::Real, k::Kernel, gpc::GPC) = GP(ConstMean(m), k, gpc)
GP(k::Kernel, gpc::GPC) = GP(ZeroMean(), k, gpc)

mean_vector(f::GP, x::AV) = ew(f.m, x)

cov(f::GP, x::AV) = pw(f.k, x)
cov_diag(f::GP, x::AV) = ew(f.k, x)

cov(f::GP, x::AV, x′::AV) = pw(f.k, x, x′)
cov_diag(f::GP, x::AV, x′::AV) = ew(f.k, x, x′)

function cov(f::GP, f′::GP, x::AV, x′::AV)
    return f === f′ ? cov(f, x, x′) : zeros(length(x), length(x′))
end
function cov_diag(f::GP, f′::GP, x::AV, x′::AV)
    return f === f′ ? cov_diag(f, x, x′) : zeros(length(x))
end
