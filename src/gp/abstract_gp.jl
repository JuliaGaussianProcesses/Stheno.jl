import Base: eachindex, rand
import Distributions: logpdf, ContinuousMultivariateDistribution
export GP, GPC, kernel, rand, logpdf, elbo, diag_cov, diag_std, marginals, mean_vec

# Pre-0.7 hack.
permutedims(v::Vector) = reshape(v, 1, length(v))

# A collection of GPs (GPC == "GP Collection"). Used to keep track of internals.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

# Supertype for all GPs.
abstract type AbstractGaussianProcess <: ContinuousMultivariateDistribution end
const AbstractGP = AbstractGaussianProcess

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
GP(μ::Tμ, k::Tk, gpc::GPC) where {Tμ, Tk} = GP{Tμ, Tk}(μ, k, gpc)
GP(k::Kernel, gpc::GPC) = GP(ZeroMean{Float64}(), k, gpc)
function GP(args...)
    μ, k, gpc = μ_p′(args...), k_p′(args...), get_check_gpc(args...)
    return GP{typeof(μ), typeof(k)}(args, μ, k, gpc)
end
show(io::IO, gp::GP) = print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)))")
==(f::GP, g::GP) = (f.μ == g.μ) && (f.k == g.k)
length(f::GP) = length(f.μ)
eachindex(f::AbstractGP) = eachindex(f.μ)

# Conversion and promotion of non-GPs to GPs.
promote(f::AbstractGP, x::Union{Real, Function}) = (f, convert(GP, x, f.gpc))
promote(x::Union{Real, Function}, f::AbstractGP) = reverse(promote(f, x))
function convert(::Type{<:AbstractGP}, x::Real, gpc::GPC)
    return GP(ConstantMean(x), ZeroKernel{Float64}(), gpc)
end
function convert(::Type{<:AbstractGP}, f::Function, gpc::GPC)
    return GP(CustomMean(f), ZeroKernel{Float64}(), gpc)
end

"""
    mean(f::GP)

The mean function of `f`.
"""
mean(f::AbstractGP) = f.μ
mean_vec(f::AbstractGP) = AbstractVector(mean(f))

"""
    kernel(f::Union{Real, Function})
    kernel(f::AbstractGP)
    kernel(f::Union{Real, Function}, g::AbstractGP)
    kernel(f::AbstractGP, g::Union{Real, Function})
    kernel(fa::AbstractGP, fb::AbstractGP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned. Also, `kernel(f) === kernel(f, f)`.
"""
kernel(f::AbstractGP) = f.k
function kernel(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    if fa === fb
        return kernel(fa)
    elseif fa.args == nothing && fa.n > fb.n || fb.args == nothing && fb.n > fa.n
        return ZeroKernel{Float64}()
    elseif fa.n > fb.n
        return k_p′p(fa.args..., fb)
    else
        return k_pp′(fa, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel{Float64}()
kernel(::Union{Real, Function}, ::AbstractGP) = ZeroKernel{Float64}()
kernel(::AbstractGP, ::Union{Real, Function}) = ZeroKernel{Float64}()

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

cov(f::AbstractGP) = AbstractMatrix(kernel(f))
marginal_cov(f::AbstractGP) = binary_obswise(kernel(f), eachindex(f))
marginal_std(f::AbstractGP) = sqrt.(marginal_cov(f))
marginals(f::AbstractGP) = (mean(f), marginal_std(f))

xcov(f::AbstractGP, g::AbstractGP) = AbstractMatrix(kernel(f, g))

"""
    rand(rng::AbstractRNG, f::AbstractGP, N::Int=1)

Obtain `N` independent samples from the GP `f` at using `rng`. Requires `length(f) < ∞`.
"""
function rand(rng::AbstractRNG, f::AbstractGP, N::Int)
    return mean_vec(f) .+ chol(cov(f))' * randn(rng, length(f), N)
end
rand(rng::AbstractRNG, f::AbstractGP) = vec(rand(rng, f, 1))

"""
    logpdf(f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real})

Returns the log probability density observing the assignments `a` jointly.
"""
function logpdf(f::AbstractGP, y::AbstractVector{<:Real})
    μ, Σ = mean_vec(f), cov(f)
    return -0.5 * (length(y) * log(2π) + logdet(Σ) + Xt_invA_X(Σ, y - μ))
end

"""
    elbo(f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real}, u::AV{<:GP}, Z::AV{<:AVM}, σ::Real)

Compute the Titsias-ELBO.
"""
function elbo(f::AbstractGP, y::AV{<:Real}, u::AbstractGP, σ::Real)
    Γ = (chol(cov(u))' \ xcov(u, f)) ./ σ
    Ω, δ = LazyPDMat(Γ * Γ' + I, 0), y - mean_vec(f)
    return -0.5 * (length(y) * log(2π * σ^2) + logdet(Ω) - sum(abs2, Γ) +
        (sum(abs2, δ) - sum(abs2, chol(Ω)' \ (Γ * δ)) + sum(marginal_cov(f))) / σ^2)
end
