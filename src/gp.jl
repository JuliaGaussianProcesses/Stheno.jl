import Base: mean
export mean, kernel, GP, observe!, predict

struct GPCollection
    gps::Vector{GP}
    k_x::ObjectIdDict
end

"""
    GP

A Gaussian Process object - specified by a mean function `μ`, a kernel `k` and cross-kernels
found in the dictionary `k_x` (note that cross-kernels are computed lazily). Also required
is the operation used to construct it, `f`, and the arguments to that operation, `args`.
A `GP` which is constructed independently of any other `GP` objects will have the `f = GP`
and `args = nothing`.
"""
struct GP
    f::Any
    args::Any
    μ::Any
    k::Any
    gpc::GPCollection
end

"""
    GP(μ, k)

Construct a Gaussian Process from a mean function and kernel.
"""
GP(μ, k, gpc::GPCollection) = GP(GP, nothing, μ, k, gpc)

"""
    mean(gp::GP)

Get the mean function of the `GP` `gp`.
"""
@inline mean(gp::GP) = gp.μ

"""
    kernel(f::GP)
    kernel(fa::GP, fb::GP)

Get the cross-covariance function between `GP`s `fa` and `fb`. If only a single `GP` is
provided, the "marginal" covariance function associated with this `GP` is returned.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    return fa === fb ? fa.k : fa.gpc.k_x[(fa, fb)]
end

k_pp′(::Any, ::GP, ::GP) = throw(error("Oops, k_pp′ not implemented."))
k_p′p(::Any, ::GP, ::GP) = throw(error("Oops, k_p′p not implemented."))

function instantiate_gp(op, args...)
    
    gpc = f_new.gpc
    for 
end

# """
#     lml(gp::GP, x::AbstractVector, y::AbstractVector)

# Compute the log marginal probability of observing `y` at `x` under `gp`.
# """
# lml(gp::GP, x::AbstractVector, y::AbstractVector) =
#     lpdf(Normal(mean(gp)(x), cov(kernel(gp), x)), y)

# """
#     sample(rng::AbstractRNG, gp::GP, x::AbstractVector, N::Int=1)

# Sample from the joint distribution that `gp` induces over the function values at input
# locations `x`.
# """
# sample(rng::AbstractRNG, gp::GP, x::AbstractVector, N::Int=1) =
#     sample(rng, Normal(mean(gp).(x), cov(kernel(gp), x)), N)

# """
#     sample(rng::AbstractRNG, pairs::Vector)

# Sample from the joint distribution over each `GP` in `pairs`. Returns a vector containing
# samples from each process provided.
# """
# function sample(rng::AbstractRNG, pairs::Vector{Tuple{GP, Tx}}) where Tx<:Vector
#     cum_lengths = vcat(0, cumsum(map(pair->length(pair[2]), pairs)))
#     μ = vcat(map(pair->mean(pair[1]).(pair[2]), pairs)...)
#     f_all = sample(rng, Normal(μ, cov(pairs...)))
#     return [view(f_all, (cum_lengths[n]+1):cum_lengths[n+1]) for n in eachindex(pairs)]
# end

# """
#     observe(gp::GP, x::Vector, f::Vector{Float64})

# Observe that the value of the `GP` `gp` is `f` at `x`.
# """
# function observe!(gp::GP, x::Vector, f::Vector{Float64})
#     append!(gp.joint.obs, (gp.idx, x, f))
#     return nothing
# end
