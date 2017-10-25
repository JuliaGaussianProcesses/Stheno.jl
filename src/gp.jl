import Base: mean, show, cov, chol, eachindex
export mean, mean_vector, kernel, GP, Normal, GPC, condition!, predict, lpdf, sample, dims

const RealVector = AbstractVector{<:Real}

struct GPC
    gps::Set{Any}
    k_x::ObjectIdDict
    obs::ObjectIdDict
    GPC() = new(Set{Any}(), ObjectIdDict(), ObjectIdDict())
end

"""
Supertype for all GP objects - infinite (GP) and finite (Normal).
"""
abstract type AbstractGP end

"""
    GP

A Gaussian Process object - specified by a mean function `μ`, a kernel `k` and cross-kernels
found in the dictionary `k_x` (note that cross-kernels are computed lazily). Also required
is the operation used to construct it, `f`, and the arguments to that operation, `args`.
A `GP` which is constructed independently of any other `GP` objects will have the `f = GP`
and `args = nothing`.
"""
struct GP <: AbstractGP
    f::Any
    args::Any
    μ::Any
    k::Any
    gpc::GPC
    function GP(op, args...)
        gpc = get_check_gpc(op, args...)
        new_gp = new(op, args, μ_p′(op, args...), k_p′(op, args...), gpc)
        populate_gpc!(new_gp, gpc, op)
        return new_gp
    end
    function GP(μ, k, gpc::GPC)
        gp = new(GP, nothing, μ, k, gpc)
        push!(gpc.gps, gp)
        return gp
    end
end

function show(io::IO, gp::GP)
    print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)) f=($(gp.f))")
end

"""
    Normal{Tμ<:AbstractVector{<:Real}, TΣ<:AbstractPDMat}

Generic multivariate Normal distribution.
"""
struct Normal <: AbstractGP
    f::Any
    args::Any
    μ::Any
    k::Any
    D::Int
    gpc::GPC
    function Normal(μ, k, dims::Int, gpc::GPC)
        gp = new(Normal, nothing, μ, k, dims, gpc)
        push!(gpc.gps, gp)
        return gp
    end
    function Normal(op, args...)
        gpc = get_check_gpc(op, args...)
        new_gp = new(op, args, μ_p′(op, args...), k_p′(op, args...), dims(op, args...), gpc)
        populate_gpc!(new_gp, gpc, op)
        return new_gp
    end
end

function populate_gpc!(new_gp, gpc, op)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(op, new_gp, gp)
        gpc.k_x[(gp, new_gp)] = k_pp′(op, gp, new_gp)
    end
    push!(gpc.gps, new_gp)
end

@inline dims(d::Normal) = d.D
@inline eachindex(f::Normal) = 1:dims(f)

mean(f::AbstractGP) = f.μ
mean_vector(f::Normal) = mean(f).(eachindex(f))
mean_vector(f::Vector{Normal}) = vcat(mean_vector.(f)...)

"""
    kernel(f::AbstractGP)
    kernel(fa::AbstractGP, fb::AbstractGP)

Get the cross-covariance function between `AbstractGP`s `fa` and `fb`. If only a single
`AbstractGP` is provided, the "marginal" covariance function associated with this
`AbstractGP` is returned.
"""
kernel(f::AbstractGP) = f.k
function kernel(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    return fa === fb ? fa.k :
        (fa, fb) in keys(fa.gpc.k_x) ?
            fa.gpc.k_x[(fa, fb)] :
            Constant(0.0)
end

k_pp′(::Any, ::AbstractGP, ::AbstractGP) = throw(error("Oops, k_pp′ not implemented."))
k_p′p(::Any, ::AbstractGP, ::AbstractGP) = throw(error("Oops, k_p′p not implemented."))

function get_check_gpc(args...)
    id = findfirst(map(arg->arg isa AbstractGP, args))
    gpc = args[id].gpc
    @assert all([!(arg isa AbstractGP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    lpdf(d::Normal, f::AbstractVector{<:Real})

Returns the log probability density of `f` under `d`.
"""
lpdf(d::Normal, f::RealVector) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), f .- mean(d).(eachindex(d))))

"""
    sample(rng::AbstractRNG, d::Normal, N::Int=1)

Take `N` samples from `d` using random number generator `rng` (not optional).
"""
sample(rng::AbstractRNG, d::Normal, N::Int) =
    mean(d).(eachindex(d)) .+ chol(cov(d)).'randn(rng, dims(d), N)
sample(rng::AbstractRNG, d::Normal) =
    mean(d).(eachindex(d)) .+ chol(cov(d)).'randn(rng, dims(d))

"""
    sample(rng::AbstractRNG, ds::Vector{Normal}, N::Int=1)

Sample jointly from multiple correlated Normal distributions.
"""
function sample(rng::AbstractRNG, ds::Vector{Normal}, N::Int)
    lin_sample = mean_vector(ds) .+ chol(cov(ds)).'randn(rng, sum(dims.(ds)), N)
    srt, fin = vcat(1, cumsum(dims.(ds))[1:end-1] .+ 1), cumsum(dims.(ds))
    return broadcast((srt, fin)->lin_sample[srt:fin, :], srt, fin)
end
function sample(rng::AbstractRNG, ds::Vector{Normal})
    lin_sample = mean_vector(ds) .+ chol(cov(ds)).'randn(rng, sum(dims.(ds)))
    srt, fin = vcat(1, cumsum(dims.(ds))[1:end-1] .+ 1), cumsum(dims.(ds))
    return broadcast((srt, fin)->lin_sample[srt:fin], srt, fin)
end
