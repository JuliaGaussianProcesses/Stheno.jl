import Base: +, length

"""
    GP

A marginal Gaussian Process object. Points to a JointGP object which contains mean and
covariance function information.
"""
struct GP
    idx::Int
    joint::JointGP
end

"""
    JointKernel
"""
struct JointKernel
    k::Vector{Vector{Any}}
end
getindex(t::JointKernel, p::Int, q::Int) = p > q ? t.xcov[q][p] : t.xcov[p][q]

"""
    JointGP

A tape-like object which keeps track of the joint distribution over the GPs.
"""
struct JointGP
    gps::Vector{GP}
    μ::Vector{Any}
    k::JointKernel
end
length(t::JointGP) = length(t.gps)

"""
    mean(gp::GP)
    mean(gp::JointGP, n::Int)

Return the mean function of the `n^{th}` `GP` in a `JointGP`, or the mean function of a
marginal `GP`.
"""
mean(gp::JointGP, n::Int) = gp.μ[n]
mean(gp::JointGP) = gp.μ
mean(gp::GP) = mean(gp.joint, gp.idx)

"""
    kernel(gp::GP)
    kernel(gp::JointGP)
    kernel(gp::JointGP, p::Int, q::Int)

Return the kernel of the `(p,q)^{th}` `GP` in a `JointGP`, or the kernel of a marginal `GP`.
If no indices are provided and a `JointGP` is provided, then the `JointKernel` object is
returned.
"""
kernel(gp::GP) = gp.joint[gp.idx, gp.idx]
kernel(gp::JointGP) = gp.k
kernel(gp::JointGP, p::Int, q::Int) = kernel(gp)[p, q]

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
"""
function +(p1::GP, p2::GP)

    # Make sure the JointKernel is common to both processes and extract it.
    p1.k != p2.k && throw(error("Tapes don't match, can't add GPs."))
    joint_kernel = p1.k

    # Compute the cross-kernels between this process and each other process + itself.
    p, q, N = p1.p, p2.p, length(joint_kernel.gps)
    ks = Vector{Kernel}(N + 1)
    for t in 1:N
        ks[p] = joint_kernel[t, p] + joint_kernel[t, q]
    end
    ks[end] = joint_kernel[p, p] + joint_kernel[q, q] + 2 * joint_kernel[p, q]
    push!(joint_kernel.xcov, ks)

    # Create new GP, append the JointKernel, and return the new GP.
    p_new = GP(x->p1.μ(x) + p2.μ(x), N + 1, joint_kernel)
    push!(joint_kernel.gps, p_new)
    return p_new
end

# NEED TO DEFINE THE ZERO KERNEL! AND THE CLOSURE NEEDS TO BE WORKED OUT, BECAUSE THE NEW
# PROCESS WILL INHERIT COVARIANCE PROPERTIES FROM PREVIOUS PROCESSES!

# Also figure out how the bloody hell to get Nabla to work with this properly. Essentially
# we need sensible functionality for quite a number of types (Symettric, StidedPDMat in
# particular, but other things will be necessary. Now that we've dropped the requirement that
# typeof(X̄) == typeof(X), we can do much more sophisticated things, and implement custom
# behaviour where necessary. We will require more sophisticated implementations for kernel
# sensitivities once we move to the multi-dimensional setting and broadcasts no longer work
# as intended).
