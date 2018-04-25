import Base: eachindex
export GP, GPC, kernel, rand, logpdf, elbo, marginal_std

# A collection of GPs (GPC == "GP Collection"). Used to keep track of internals.
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
==(f::GP, g::GP) = (f.μ == g.μ) && (f.k == g.k)
length(f::GP) = length(f.μ)
eachindex(f::GP) = eachindex(f.μ)

# Conversion and promotion of non-GPs to GPs.
promote(f::GP, x::Union{Real, Function}) = (f, convert(GP, x, f.gpc))
promote(x::Union{Real, Function}, f::GP) = reverse(promote(f, x))
convert(::Type{GP}, x::Real, gpc::GPC) = GP(ConstantMean(x), ZeroKernel{Float64}(), gpc)
convert(::Type{GP}, f::Function, gpc::GPC) = GP(CustomMean(f), ZeroKernel{Float64}(), gpc)

"""
    kernel(f::Union{Real, Function})
    kernel(f::GP)
    kernel(f::Union{Real, Function}, g::GP)
    kernel(f::GP, g::Union{Real, Function})
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned. Also, `kernel(f) === kernel(f, f)`.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    if fa === fb
        return kernel(fa)
    elseif fa.args == nothing && fa.n > fb.n || fb.args == nothing && fb.n > fa.n
        return ZeroKernel{Float64}()
    elseif fa.n > fb.n
        return k_p′p(fb, fa.f, fa.args...)
    else
        return k_pp′(fa, fb.f, fb.args...)
    end
end
kernel(::Union{Real, Function}) = ZeroKernel{Float64}()
kernel(::Union{Real, Function}, ::GP) = ZeroKernel{Float64}()
kernel(::GP, ::Union{Real, Function}) = ZeroKernel{Float64}()

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    mean(f::GP)

The mean function of `f`.
"""
mean(f::GP) = f.μ

"""
    mean(f::GP, X::AVM)

The mean function of `f` evaluated at `X` is a vector whose length is `size(X, 1)`.
"""
mean(f::GP, X::AVM) = mean(f.μ, X)


"""
    cov(f::GP, X::AVM)

The covariance of `f` evaluated at `X` is an `size(X, 1) x size(X, 1)` `LazyPDMat`.
"""
cov(f::GP, X::AVM) = cov(f.k, X)

"""
    marginal_cov(f::GP, X::AVM)

Yields the same result as `diag(cov(f, X))`, but with `O(size(X, 1))` complexity.
"""
marginal_cov(f::GP, X::AVM) = marginal_cov(f.k, X)

"""
    xcov(f::GP, f′::GP, X::AVM, X′::AVM)


The cross-covariance between `f` at `X` and `f′` at `X′`.
"""
xcov(f::GP, f′::GP, X::AVM, X′::AVM) = xcov(kernel(f, f′), X, X′)

"""
    xcov(f::GP, f′::GP, X::AVM)

The cross-covariance between `f` at `X` and `f′` at `X`.
"""
xcov(f::GP, f′::GP, X::AVM) = xcov(f, f′, X, X)

"""
    xcov(f::GP, X::AVM, X′::AVM)

The cross-covariance between `f` at `X` and `f` at `X′`.
"""
xcov(f::GP, X::AVM, X′::AVM) = xcov(f.k, X, X′)

# Definitions of `mean`, `cov`, `xcov`, and `marginal_cov` for vectors of `GP`s.
mean(f::AV{<:GP}, X::AV{<:AVM}) = mean(CatMean(f), X)
cov(f::AV{<:GP}, X::AV{<:AVM}) = cov(CatKernel(kernel.(f), kernel.(f, permutedims(f))), X)
xcov(f::AV{<:GP}, f′::AV{<:GP}, X::AV{<:AVM}, X′::AV{<:AVM}) =
    xcov(CatCrossKernel(kernel.(f, permutedims(f′))), X, X′)
marginal_cov(f::AV{<:GP}, X::AV{<:AVM}) = vcat(marginal_cov.(f, X)...)

"""
    marginal_std(f::Union{GP, AV{<:GP}}, X::AVM)

Broadcasts `sqrt` over `marginal_cov(f, X)`.
"""
marginal_std(f::Union{GP, AV{<:GP}}, X::AVM) = sqrt.(marginal_cov(f, X))

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation
    f::GP
    y::Vector
end
←(f, y) = Observation(f, y)

"""
    rand(rng::AbstractRNG, f::GP, X::AM, N::Int=1)

Obtain `N` independent samples from the GP `f` at `X` using `rng`.
"""
rand(rng::AbstractRNG, f::AV{<:GP}, X::AV{<:AVM}, N::Int) =
    mean(f, X) .+ chol(cov(f, X))' * randn(rng, sum(size.(X, 1)), N)
rand(rng::AbstractRNG, f::AV{<:GP}, X::AV{<:AVM}) = vec(rand(rng, f, X, 1))
rand(rng::AbstractRNG, f::GP, X::AVM, N::Int) = rand(rng, [f], [X], N)
rand(rng::AbstractRNG, f::GP, X::AVM) = rand(rng, [f], [X])

"""
    logpdf(f::AV{<:GP}, X::AV{<:AVM}, y::AV{<:AV})

Returns the log probability density observing the assignments `a` jointly.
"""
function logpdf(f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real})
    μ, Σ = mean(f, X), cov(f, X)
    return -0.5 * (length(y) * log(2π) + logdet(Σ) + Xt_invA_X(Σ, y - μ))
end
logpdf(f::GP, X::AVM, y::AV{<:Real}) = logpdf([f], [X], BlockVector([y]))

"""
    elbo(f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real}, u::AV{<:GP}, Z::AV{<:AVM}, σ::Real)

Compute the Titsias-ELBO. Doesn't currently work because I've not implemented `vcat` for
`GP`s at all. I've also not tested `logpdf` yet, so I should probably do that...
"""
function elbo(f::AV{<:GP}, X::AV{<:AVM}, y::BlockVector{<:Real}, u::AV{<:GP}, Z::AV{<:AVM}, σ::Real)
    Γ = (chol(cov(u, Z))' \ xcov(u, f, Z, X)) ./ σ
    Ω, δ = LazyPDMat(Γ * Γ' + I, 0), y - mean(f, X)
    return -0.5 * (length(y) * log(2π * σ^2) + logdet(Ω) - sum(abs2, Γ) +
        (sum(abs2, δ) - sum(abs2, chol(Ω)' \ (Γ * δ)) + sum(marginal_cov(f, X))) / σ^2)
end
