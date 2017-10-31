import Base: mean, show, cov, chol, eachindex
export mean, mean_vector, kernel, GP, GPC, condition!, predict, lpdf, sample, dims

# A collection of GPs (GPC == "GP Collection"). Primarily used to track cross-kernels.
struct GPC
    gps::Set{Any}
    k_x::ObjectIdDict
    GPC() = new(Set{Any}(), ObjectIdDict())
end

"""
    GP{Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tk<:Kernel}
    f::Any
    args::Any
    μ::Any
    k::Tk
    gpc::GPC
    GP{Tk}(f, args, μ, k::Tk, gpc::GPC) where Tk<:Kernel = new{Tk}(f, args, μ, k, gpc)
    function GP{Tk}(μ, k::Tk, gpc::GPC) where Tk<:Kernel
        gp = new(GP, nothing, μ, k, gpc)
        push!(gpc.gps, gp)
        return gp
    end
end
function GP(op, args...)
    gpc = get_check_gpc(op, args...)
    k = k_p′(op, args...)
    new_gp = GP{typeof(k)}(op, args, μ_p′(op, args...), k, gpc)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(op, new_gp, gp)
        gpc.k_x[(gp, new_gp)] = k_pp′(op, gp, new_gp)
    end
    push!(gpc.gps, new_gp)
    return new_gp
end
GP(μ, k::Tk, gpc::GPC) where Tk = GP{Tk}(μ, k, gpc)
show(io::IO, gp::GP) = print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)) f=($(gp.f))")

@inline dims(d::GP) = dims(kernel(d))
@inline eachindex(f::GP) = 1:dims(f)

mean(f::GP) = f.μ
mean_vector(f::GP) = mean(f).(eachindex(f))
mean_vector(f::Vector) = vcat(mean_vector.(f)...)

"""
    kernel(f::GP)
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and `kernel(f) == kernel(f, f)`.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    return fa === fb ?
        fa.k :
        (fa, fb) in keys(fa.gpc.k_x) ? fa.gpc.k_x[(fa, fb)] : Constant(0.0)
end

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    lpdf(d::GP, f::AbstractVector{<:Real})

Returns the log probability density of `f` under `d`. Dims `d` must be finite.
"""
lpdf(d::GP, f::AbstractVector{<:Real}) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), f .- mean_vector(d)))

"""
    sample(rng::AbstractRNG, d::Union{GP, Vector}, N::Int=1)

Sample jointly from a single / multiple finite-dimensional GPs.
"""
function sample(rng::AbstractRNG, ds::Vector, N::Int)
    lin_sample = mean_vector(ds) .+ chol(cov(ds)).'randn(rng, sum(dims.(ds)), N)
    srt, fin = vcat(1, cumsum(dims.(ds))[1:end-1] .+ 1), cumsum(dims.(ds))
    return broadcast((srt, fin)->lin_sample[srt:fin, :], srt, fin)
end
sample(rng::AbstractRNG, ds::Vector) = reshape.(sample(rng, ds, 1), dims.(ds))
sample(rng::AbstractRNG, d::GP, N::Int) = sample(rng, [d], N)[1]
sample(rng::AbstractRNG, d::GP) = sample(rng, [d])[1]
