export GP

abstract type AbstractGP end

# A collection of GPs (GPC == "GP Collection"). Used to keep track of GPs.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

@nograd GPC

next_index(gpc::GPC) = gpc.n + 1



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
GP(m::Tm, k::Tk, gpc::GPC) where {Tm<:MeanFunction, Tk<:CrossKernel} = GP{Tm, Tk}(m, k, gpc)

GP(m::Real, k::Kernel, gpc::GPC) = GP(ConstMean(m), k, gpc)
GP(k::Kernel, gpc::GPC) = GP(ZeroMean(), k, gpc)

mean_vector(f::GP, x::AV) = ew(f.m, x)
cov(f::GP, x::AV) = pw(f.k, x)
cov(f::GP, x::AV, x′::AV) = pw(f.k, x, x′)
cov_diag(f::GP, x::AV) = ew(f.k, x)

index(f::GP) = f.n
independent(f::GP, f′::GP) = (f.gpc === f′.gpc) && (f.n != f′.n)

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
cov(f::CompositeGP, x::AV, x′::AV) = cov(f.args, x, x′)
cov_diag(f::CompositeGP, x::AV) = cov_diag(f.args, x)

is_atomic(f::GP) = true
is_atomic(f::AbstractGP) = false

# Compute the cross-covariance matrix between `f` at `x` and `f′` at `x′`.
function xcov(f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov(f, x, x′)
    elseif is_atomic(f) && f.n > f′.n || is_atomic(f′) && f′.n > f.n
        return zeros(length(x), length(x′))
    elseif f.n > f′.n
        return xcov(f.args, f′, x, x′)
    else
        return xcov(f, f′.args, x, x′)
    end
end

xcov(f::AbstractGP, f′::AbstractGP, x::AV) = xcov(f, f′, x, x)

function xcov_diag(f::AbstractGP, f′::AbstractGP, x::AV)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov_diag(f, x)
    elseif is_atomic(f) && f.n > f′.n || is_atomic(f′) && f′.n > f.n
        return zeros(length(x))
    elseif f.n > f′.n
        return xcov_diag(f.args, f′, x)
    else
        return xcov_diag(f, f′.args, x)
    end
end

index(f::CompositeGP) = f.n
