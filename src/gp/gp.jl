export GP

abstract type AbstractGP end

const AGP = AbstractGP

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

mean_vector(f::GP, x::AV) = ew(f.m, x)
cov_mat(f::GP, x::AV) = pw(f.k, x)
cov_mat(f::GP, x::AV, x′::AV) = pw(f.k, x, x′)
cov_mat_diag(f::GP, x::AV) = ew(f.k, x)

index(f::GP) = f.n



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
cov_mat(f::CompositeGP, x::AV) = cov_mat(f.args, x)
cov_mat_diag(f::CompositeGP, x::AV) = cov_mat_diag(f.args, x)
cov_mat(f::CompositeGP, x::AV, x′::AV) = cov_mat(f.args, x, x′)

index(f::CompositeGP) = f.n



#
# Logic to compute the cross-covariance between a pair of Gaussian processes.
#

is_atomic(f::GP) = true
is_atomic(f::AbstractGP) = false

# Compute the cross-covariance matrix between `f` at `x` and `f′` at `x′`.
function xcov_mat(f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    @assert f.gpc === f′.gpc
    if f.n === f′.n
        return cov_mat(f, x, x′)
    elseif is_atomic(f) && f.n > f′.n || is_atomic(f′) && f′.n > f.n
        return zeros(length(x), length(x′))
    elseif f.n > f′.n
        return xcov_mat(f.args, f′, x, x′)
    else
        return xcov_mat(f, f′.args, x, x′)
    end
end

xcov_mat(f::AbstractGP, f′::AbstractGP, x::AV) = xcov_mat(f, f′, x, x)
