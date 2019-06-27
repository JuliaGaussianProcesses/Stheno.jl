export GP, mean, kernel

# A collection of GPs (GPC == "GP Collection"). Used to keep track of internals.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

@nograd GPC

"""
    GP{Tμ<:MeanFunction, Tk<:CrossKernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tμ<:MeanFunction, Tk<:CrossKernel}
    args::Any
    μ::Tμ
    k::Tk
    n::Int
    gpc::GPC
    function GP{Tμ, Tk}(args, μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk<:CrossKernel}
        gp = new{Tμ, Tk}(args, μ, k, Zygote.dropgrad(gpc.n), Zygote.dropgrad(gpc))
        gpc.n += 1
        return gp
    end
    GP{Tμ, Tk}(μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk} = GP{Tμ, Tk}((), μ, k, gpc)
end
GP(μ::Tμ, k::Tk, gpc::GPC) where {Tμ<:MeanFunction, Tk<:CrossKernel} = GP{Tμ, Tk}(μ, k, gpc)

# GP initialised with a constant mean. Zero and one are specially handled.
function GP(m::Real, k::CrossKernel, gpc::GPC)
    if iszero(m)
        return GP(k, gpc)
    elseif isone(m)
        return GP(OneMean(), k, gpc)
    else
        return GP(ConstMean(m), k, gpc)
    end
end
GP(m, k::CrossKernel, gpc::GPC) = GP(CustomMean(m), k, gpc)
GP(k::CrossKernel, gpc::GPC) = GP(ZeroMean(), k, gpc)
function GP(gpc::GPC, args...)
    μ, k = μ_p′(args...), k_p′(args...)
    return GP{typeof(μ), typeof(k)}(args, μ, k, gpc)
end

mean(f::GP) = f.μ

"""
    kernel(f::Union{Real, Function})
    kernel(f::GP)
    kernel(f::Union{Real, Function}, g::GP)
    kernel(f::GP, g::Union{Real, Function})
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned. Also, `kernel(f) === kernel(f, f)`.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    if fa === fb
        return kernel(fa)
    elseif fa.args == () && fa.n > fb.n || fb.args == () && fb.n > fa.n
        return ZeroKernel()
    elseif fa.n > fb.n
        return k_p′p(fa.args..., fb)
    else
        return k_pp′(fa, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel()
kernel(::Union{Real, Function}, ::GP) = ZeroKernel()
kernel(::GP, ::Union{Real, Function}) = ZeroKernel()
