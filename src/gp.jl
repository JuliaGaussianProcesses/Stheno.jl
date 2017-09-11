import Base: +, length, getindex, push!, mean, size
export mean, kernel, GP

"""
    KernelCollection

An upper triangular matrix represented as a vector of vectors, the pth of whom's length is
p, which will contain kernels. This is an internal data structure, that the user of the
package should not need to interact with directly.
"""
struct KernelCollection
    k::Vector{<:Vector{<:Any}}
    KernelCollection(k::Vector{<:Vector{<:Any}}) = new(k)
    KernelCollection() = new(Vector{Vector{Any}}())
end
getindex(t::KernelCollection, p::Int, q::Int) = p > q ? t.k[p][q] : t.k[q][p]
push!(t::KernelCollection, x::Vector{<:Any}) = (push!(t.k, x); t)
size(t::KernelCollection) = (length(t.k), length(t.k))
size(t::KernelCollection, d::Int) =
    d < 1 ? throw(error("arraysize: dim out of bounds.")) : (d < 3 ? length(t.k) : 1)

"""
    GPCollection

A a representation of collection of Gaussian Processes. This is an internal data structure
that the user of the package should not need to interact with directly.
"""
struct GPCollection
    μ::Vector{<:Any}
    k::KernelCollection
    GPCollection() = new(Vector{Any}(), KernelCollection())
    function GPCollection(μ::Vector{<:Any}, k::KernelCollection)
        length(μ) != size(k, 1) && throw(ArgumentError("μ and k have different sizes."))
        return new(μ, k)
    end
end

"""
    length(gpc::GPCollection)

The number of Gaussian Processes represented by `gpc`.
"""
length(gpc::GPCollection) = length(gpc.μ)

"""
    mean(gpc::GPCollection, p::Int)

Mean function of the `p`th GP in `gpc`.
"""
mean(gpc::GPCollection, p::Int) = gpc.μ[p]

"""
    kernel(gpc::GPCollection, p::Int, q::Int)
    kernel(gpc::GPCollection, p::Int)

Get the cross-covariance between the `p`th and `q`th processes in `gpc`. If only a single
index `p` is provided, or `p == q`, then the covariance function of the `p`th process in
`gpc` is returned.
"""
kernel(gpc::GPCollection, p::Int, q::Int) = gpc.k[p, q]
kernel(gpc::GPCollection, p::Int) = kernel(gpc, p, p)

"""
    append_indep!(gp::GPCollection, μ, k::Kernel)

Append an independent GP onto `gp`, with mean function `μ` and kernel `k`.
"""
function append_indep!(gpc::GPCollection, μ, k::Kernel)
    len = length(gpc)
    push!(gpc.μ, μ)
    push!(gpc.k, vcat(fill(Constant(0.0), len), k))
    return gpc
end

"""
    GP

A marginal Gaussian Process object. Points to a GPCollection object which contains mean and
covariance function information.
"""
struct GP
    idx::Int
    joint::GPCollection
    GP(idx::Int, jgp::GPCollection) = new(idx, jgp)
    function GP(jgp::GPCollection, μ, k::Kernel)
        jgp = append_indep!(jgp, μ, k)
        return new(length(jgp), jgp)
    end
end

"""
    mean(gp::GP)

Get the mean function of the marginal `GP` `gp`.
"""
mean(gp::GP) = mean(gp.joint, gp.idx)

"""
    kernel(gp1::GP, gp2::GP)
    kernel(gp::GP)

Get the cross-covariance function between `GP`s `gp1` and `gp2`. If only a single `GP` is
provided, the "marginal" covariance function associated with this `GP` is returned. If
`gp1` and `gp2` point towards different `GPCollection`s, an assertion will fail.
"""
@inline function kernel(gp1::GP, gp2::GP)
    @assert gp1.joint === gp2.joint
    return kernel(gp1.joint, gp1.idx, gp2.idx)
end
kernel(gp::GP) = kernel(gp.joint, gp.idx)

"""
    lml(gp::GP, x::AbstractVector, y::AbstractVector)

Compute the log marginal probability of observing `y` at `x` under `gp`.
"""
lml(gp::GP, x::AbstractVector, y::AbstractVector) =
    lpdf(Normal(mean(gp)(x), cov(kernel(gp), x)), y)

"""
    sample(rng::AbstractRNG, gp::GP, x::AbstractVector, N::Int=1)

Sample from the joint distribution that `gp` induces over the function values at input
locations `x`.
"""
sample(rng::AbstractRNG, gp::GP, x::AbstractVector, N::Int=1) =
    sample(rng, Normal(mean(gp)(x), cov(kernel(gp), x)), N)

"""
    +(p1::GP, p2::GP)

Return the process that one obtains when two GPs are added together, and compute all of the
information necessary to perform inference in the joint distribution over the GPs.
If `p1` and `p2` point towards different `GPCollection`s, an assertion will fail.
"""
function +(p1::GP, p2::GP)

    # Make sure the KernelCollection is common to both processes and extract it.
    @assert p1.joint === p2.joint
    joint, p, q = p1.joint, p1.idx, p2.idx

    # Compute and push the mean function of the new GP.
    μ = mean(joint)
    μp, μq = μ[p], μ[q]
    push!(μ, x->μp(x) + μq(x))

    # Compute + push the cross-kernels between this process and each other process + itself.
    k, p, q, N = kernel(joint), p1.idx, p2.idx, length(joint)
    ks = Vector{Kernel}(N + 1)
    for t in 1:N
        ks[p] = k[t, p] + k[t, q]
    end
    ks[end] = k[p, p] + k[q, q] + 2 * k[p, q]
    push!(k, ks)

    # Create and return the new GP.
    return GP(joint, N + 1)
end
