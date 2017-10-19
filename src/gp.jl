import Base: mean, show, cov, chol
export mean, kernel, GP, Normal, GPC, condition!, predict, lpdf, sample, dims

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
end

"""
    GP(μ, k, gpc::GPC)

Construct a Gaussian Process from a mean function and kernel.
"""
function GP(μ, k, gpc::GPC)
    gp = GP(GP, nothing, μ, k, gpc)
    push!(gpc.gps, gp)
    return gp
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
end

@inline mean(gp::AbstractGP) = gp.μ
@inline dims(d::Normal) = d.D

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

function instantiate_gp(op, args...)
    gpc = get_check_gpc(op, args...)
    new_gp = GP(op, args, μ_p′(op, args...), k_p′(op, args...), gpc)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(op, new_gp, gp)
        gpc.k_x[(gp, new_gp)] = k_pp′(op, gp, new_gp)
    end
    push!(gpc.gps, new_gp)
    return new_gp
end

function instantiate_normal(op, args...)
    gpc = get_check_gpc(op, args...)
    new_gp = Normal(op, args, μ_p′(op, args...), k_p′(op, args...), dims(op, args...), gpc)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(op, new_gp, gp)
        gpc.k_x[(gp, new_gp)] = k_pp′(op, gp, new_gp)
    end
    push!(gpc.gps, new_gp)
    return new_gp
end

"""
    lpdf(d::Normal, f::AbstractVector{<:Real})

Returns the log probability density of `f` under `d`.
"""
lpdf(d::Normal, f::RealVector) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), f .- mean(d).(1:dims(d))))

"""
    sample(rng::AbstractRNG, d::Normal, N::Int=1)

Take `N` samples from `d` using random number generator `rng` (not optional).
"""
sample(rng::AbstractRNG, d::Normal, N::Int=1) =
    mean(d).(1:dims(d)) .+ chol(cov(d)).'randn(rng, dims(d), N)

"""
    condition!(d::Normal, f::Vector{<:Real})

Observe that the value of the `Normal` `d` is `f`.
"""
function condition!(d::Normal, f::RealVector)
    d.gpc.obs[d] = f
    return nothing
end
