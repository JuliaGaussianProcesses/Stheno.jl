export GP, GPC, kernel, logpdf, mean_function

# A collection of GPs (GPC == "GP Collection"). Primarily used to track cross-kernels.
mutable struct GPC
    n::Int
    GPC() = new(0)
end

"""
    GP{Tμ<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tμ<:MeanFunction, Tk<:Kernel}
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
isfinite(f::GP) = isfinite(f.k)
length(f::GP) = length(f.μ)
mean_function(f::GP) = f.μ

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
        return zero_kernel(fa, fb)
    elseif fa.n > fb.n
        return k_p′p(fb, fa.f, fa.args...)
    else
        return k_pp′(fa, fb.f, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel{Float64}()
kernel(::Union{Real, Function}, ::GP) = ZeroKernel{Float64}()
kernel(::GP, ::Union{Real, Function}) = ZeroKernel{Float64}()

# Convenience function to return the correct type of Zero kernel for finite dimensional GPs.
zero_kernel(fa::GP{<:FiniteMean, <:FiniteKernel}, fb::GP{<:FiniteMean, <:FiniteKernel}) =
    FiniteCrossKernel(ZeroKernel{Float64}(), fa.k.X, fb.k.X)
zero_kernel(fa::GP{<:FiniteMean, <:FiniteKernel}, fb::GP) =
    LhsFiniteCrossKernel(ZeroKernel{Float64}(), fa.k.X)
zero_kernel(fa::GP, fb::GP{<:FiniteMean, <:FiniteKernel}) =
    RhsFiniteCrossKernel(ZeroKernel{Float64}(), fb.k.X)
zero_kernel(fa::GP, fb::GP) = ZeroKernel{Float64}()

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

mean(f::GP) = mean(f.μ)
mean(f::GP, X::AM) = mean(f.μ, X)

cov(f::GP) = cov(f.k)
cov(f::GP, X::AM) = cov(f.k, X)

xcov(f::GP, f′::GP) = xcov(kernel(f, f′))
xcov(f::GP, X::AM, X′::AM) = xcov(f.k, X, X′)
xcov(f::GP, f′::GP, X::AM, X′::AM) = xcov(kernel(f, f′), X, X′)

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
function logpdf(a::Observation...)
    f, y = vcat(map(a_->a_.f, a)...), vcat(map(a_->a_.y, a)...)
    μ, Σ = mean(f), cov(f)
    return -0.5 * (length(f) * log(2π) + logdet(Σ) + invquad(Σ, y - μ))
end

"""
    rand(rng::AbstractRNG, f::GP, N::Int=1)

Obtain `N` independent samples from the (finite-dimensional) GP `f` using `rng`.
"""
rand(rng::AbstractRNG, f::GP, N::Int) = mean(f) .+ chol(cov(f))' * randn(rng, length(f), N)
rand(rng::AbstractRNG, f::GP) = vec(rand(rng, f, 1))



# LEGACY CODE.

# """
#     logpdf(a::Vector{Observation}})

# Returns the log probability density observing the assignments `a` jointly.
# """
# function logpdf(a::Vector{Observation})
#     f, y = [c̄.f for c̄ in a], [c̄.y for c̄ in a]
#     Σ = cov(f)
#     δΣinvδ = invquad(Σ, vcat(y...) .- mean(f))
#     return -0.5 * (sum(length.(f)) * log(2π) + logdet(Σ) + δΣinvδ)
# end
# logpdf(a::Observation...) = logpdf([a...])

# function rand(rng::AbstractRNG, ds::Vector{<:GP}, N::Int)
#     μ = vcat(mean.(mean.(ds))...) # This looks ridiculous and will be fixed by issue #3.
#     lin_sample = μ .+ Transpose(chol(cov(ds))) * randn(rng, sum(length.(ds)), N)
#     @show lin_sample
#     srt, fin = vcat(1, cumsum(length.(ds))[1:end-1] .+ 1), cumsum(length.(ds))
#     return broadcast((srt, fin)->lin_sample[srt:fin, :], srt, fin)
# end
# # rand(rng::AbstractRNG, ds::Vector{<:GP}) = reshape.(rand(rng, ds, 1), length.(ds))
# rand(rng::AbstractRNG, d::GP, N::Int) = rand(rng, [d], N)[1]
# rand(rng::AbstractRNG, d::GP) = rand(rng, [d])[1]

# """
#     cov(d::Union{GP, Vector{<:GP}}, d′::Union{GP, Vector{<:GP}})

# Compute the cross-covariance between GPs (or vectors of) `d` and `d′`.
# """
# function cov(
#     d::Vector{GP{FiniteMean, FiniteKernel}},
#     d′::Vector{GP{FiniteMean, FiniteKernel}},
# )
#     return cov(broadcast((f, f′)->kernel(f, f′), d, permutedims(d′)))
# end
# cov(d::Vector{<:GP}, d′::GP) = cov(d, [d′])
# cov(d::GP, d′::Vector{<:GP}) = cov([d], d′)
# cov(d::GP, d′::GP) = cov([d], [d′])

# """
#     cov(d::Union{GP, Vector{<:GP}})

# Compute the marginal covariance matrix for GP (or vector thereof) `d`.
# """
# cov(d::Vector{<:GP}) = cov(d, d)
# cov(d::GP) = cov([d])
