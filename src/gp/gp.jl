export GP

###################### Implementation of AbstractGaussianProcess interface #################

"""
    GP{Tμ<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tμ<:MeanFunction, Tk<:Kernel} <: AbstractGaussianProcess
    args::Any
    μ::Tμ
    k::Tk
    n::Int
    gpc::GPC
    function GP{Tμ, Tk}(args, μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk<:Kernel}
        gp = new{Tμ, Tk}(args, μ, k, gpc.n, gpc)
        gpc.n += 1
        return gp
    end
    GP{Tμ, Tk}(μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk} = GP{Tμ, Tk}(nothing, μ, k, gpc)
end
GP(μ::Tμ, k::Tk, gpc::GPC) where {Tμ<:MeanFunction, Tk<:Kernel} = GP{Tμ, Tk}(μ, k, gpc)

# GP initialised with a constant mean. Zero is specially handled.
function GP(m::Real, k::Kernel, gpc::GPC)
    if isfinite(length(k))
        if iszero(m)
            return GP(zero(EmpiricalMean(Zeros(length(k)))), k, gpc)  # hack
        else
            return GP(FiniteMean(ConstantMean(m), eachindex(k)), k, gpc)
        end
    else
        if iszero(m)
            return GP(zero(ConstantMean(m)), k, gpc)  # hack
        else
            return GP(ConstantMean(m), k, gpc)
        end
    end
end
GP(m, k::Kernel, gpc::GPC) = GP(CustomMean(m), k, gpc)
GP(k::Kernel, gpc::GPC) = GP(ZeroMean{Float64}(), k, gpc)
function GP(args...)
    μ, k, gpc = μ_p′(args...), k_p′(args...), get_check_gpc(args...)
    return GP{typeof(μ), typeof(k)}(args, μ, k, gpc)
end

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

show(io::IO, gp::GP) = print(io, gp)

function Base.print(io::IO, gp::GP)
    println(io, "GP with mean:")
    println(io, Shunted(Shunt(4, ' '), gp.μ))
    println(io, "and kernel:")
    print(io, Shunted(Shunt(4, ' '), gp.k))
end

mean(f::GP) = f.μ

"""
    kernel(f::Union{Real, Function})
    kernel(f::AbstractGP)
    kernel(f::Union{Real, Function}, g::AbstractGP)
    kernel(f::AbstractGP, g::Union{Real, Function})
    kernel(fa::AbstractGP, fb::AbstractGP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned. Also, `kernel(f) === kernel(f, f)`.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    if fa === fb
        return kernel(fa)
    elseif fa.args == nothing && fa.n > fb.n || fb.args == nothing && fb.n > fa.n
        if isfinite(length(fa)) && isfinite(length(fb))
            return FiniteZeroCrossKernel(eachindex(fa), eachindex(fb))
        elseif isfinite(length(fa)) && !isfinite(length(fb))
            return LhsFiniteZeroCrossKernel(eachindex(fa))
        elseif !isfinite(length(fa)) && isfinite(length(fb))
            return RhsFiniteZeroCrossKernel(eachindex(fb))
        else # Both processes are infinite dimensional.
            return ZeroKernel{Float64}()
        end
    elseif fa.n > fb.n
        return k_p′p(fa.args..., fb)
    else
        return k_pp′(fa, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel{Float64}()
kernel(::Union{Real, Function}, ::GP) = ZeroKernel{Float64}()
kernel(::GP, ::Union{Real, Function}) = ZeroKernel{Float64}()



##################################### Syntactic Sugar#######################################

import Base: promote, convert

get_zero(p::Int) = get_zero(p, p)
function get_zero(p::Real, q::Real)
    if isfinite(p) && isfinite(q)
        return p == q ? FiniteZeroKernel(1:p) : FiniteZeroCrossKernel(p, q)
    elseif isfinite(p)
        return LhsFiniteZeroCrossKernel(1:p)
    elseif isfinite(q)
        return RhsFiniteZeroCrossKernel(1:q)
    else
        return ZeroKernel{Float64}()
    end
end

promote(f::GP, x::Union{Real, Function, MeanFunction}) = (f, convert(GP, x, f.gpc))
promote(x::Union{Real, Function, MeanFunction}, f::GP) = reverse(promote(f, x))
convert(::Type{<:GP}, x::Real, gpc::GPC) = GP(ConstantMean(x), ZeroKernel{typeof(x)}(), gpc)
convert(::Type{<:GP}, f::Function, gpc::GPC) = GP(CustomMean(f), ZeroKernel{Float64}(), gpc)
convert(::Type{<:GP}, μ::MeanFunction, gpc::GPC) = GP(μ, get_zero(length(μ)), gpc)
