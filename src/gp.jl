import LinearAlgebra: chol, transpose
export GP, GPC, kernel, logpdf

# A collection of GPs (GPC == "GP Collection"). Primarily used to track cross-kernels.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

"""
    GP{Tμ, Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tμ, Tk<:Kernel}
    f::Any
    args::Any
    μ::Tμ
    k::Tk
    n::Int
    gpc::GPC
    function GP{Tμ, Tk}(f, args, μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk<:Kernel}
        gp = new{Tμ, Tk}(f, args, μ, k, gpc.n, gpc)
        gpc.n += 1
        return gp
    end
    GP{Tμ, Tk}(μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk<:Kernel} =
        GP{Tμ, Tk}(GP, nothing, μ, k, gpc)
end
GP(μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk<:Kernel} = GP{Tμ, Tk}(μ, k, gpc)
function GP(op, args...)
    μ, k, gpc = μ_p′(op, args...), k_p′(op, args...), get_check_gpc(op, args...)
    return GP{typeof(μ), typeof(k)}(op, args, μ, k, gpc)
end
show(io::IO, gp::GP) = print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)) f=($(gp.f))")
transpose(f::GP) = f
isfinite(f::GP) = isfinite(f.k)

mean(f::GP) = f.μ
mean(f::GP, X::AM) = mean(f.μ, X)

"""
    kernel(f::Union{Real, Function})
    kernel(f::GP)
    kernel(f::Union{Real, Function}, g::GP)
    kernel(f::GP, g::Union{Real, Function})
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned.
`kernel(f) === kernel(f, f)`
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    if fa === fb
        return kernel(fa)
    elseif fa.args == nothing && fa.n > fb.n || fb.args == nothing && fb.n > fa.n
        return ZeroKernel()
    elseif fa.n > fb.n
        return k_p′p(fb, fa.f, fa.args...)
    else
        return k_pp′(fa, fb.f, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel()
kernel(::Union{Real, Function}, ::GP) = ZeroKernel()
kernel(::GP, ::Union{Real, Function}) = ZeroKernel()

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    cov(d::Union{GP, Vector{<:GP}}, d′::Union{GP, Vector{<:GP}})

Compute the cross-covariance between GPs (or vectors of) `d` and `d′`.
"""
cov(d::Vector{<:GP}, d′::Vector{<:GP}) = cov(kernel.(d, Transpose(d′)))
cov(d::Vector{<:GP}, d′::GP) = cov(d, [d′])
cov(d::GP, d′::Vector{<:GP}) = cov([d], d′)
cov(d::GP, d′::GP) = cov([d], [d′])

"""
    cov(d::Union{GP, Vector{<:GP}})

Compute the marginal covariance matrix for GP (or vector thereof) `d`.
"""
cov(d::Vector{<:GP}) = cov(d, d)
cov(d::GP) = cov([d])

# """
#     cov(d::Union{GP, Vector{<:GP}})

# Compute the marginal covariance matrix for GP (or vector thereof) `d`.
# """
# function cov(d::Vector{<:GP})
#     K = cov(kernel.(d, Transpose(d)))::Matrix{Float64}
#     K[diagind(K)] .+= __ϵ
#     LAPACK.potrf!('U', K)
#     return StridedPDMatrix(UpperTriangular(K))
# end
# cov(d::GP) = cov([d])

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value. Yields a very
pleasing syntax, along the following lines: `f(X) ← y`.
"""
struct Observation
    f::GP
    y::Vector
end
←(f, y) = Observation(f, y)


"""
    logpdf(a::Vector{Observation}})

Returns the log probability density observing the assignments `a` jointly.
"""
function logpdf(a::Vector{Observation})
    f, y = [c̄.f for c̄ in a], [c̄.y for c̄ in a]
    Σ = cov(f)
    δΣinvδ = invquad(Σ, vcat(y...) .- mean_vector(f))
    return -0.5 * (sum(dims.(f)) * log(2π) + logdet(Σ) + δΣinvδ)
end
logpdf(a::Observation...) = logpdf([a...])


"""
    rand(rng::AbstractRNG, d::Union{GP, Vector}, N::Int=1)

Sample jointly from a single / multiple finite-dimensional GPs.
"""
function rand(rng::AbstractRNG, ds::Vector{GP}, N::Int)
    lin_sample = mean_vector(ds) .+ Transpose(chol(cov(ds))) * randn(rng, sum(dims.(ds)), N)
    srt, fin = vcat(1, cumsum(dims.(ds))[1:end-1] .+ 1), cumsum(dims.(ds))
    return broadcast((srt, fin)->lin_sample[srt:fin, :], srt, fin)
end
rand(rng::AbstractRNG, ds::Vector{GP}) = reshape.(rand(rng, ds, 1), dims.(ds))
rand(rng::AbstractRNG, d::GP, N::Int) = rand(rng, Vector{GP}([d]), N)[1]
rand(rng::AbstractRNG, d::GP) = rand(rng, Vector{GP}([d]))[1]
