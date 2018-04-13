export GP, GPC, kernel, logpdf, mean_function

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

mean(f::GP, X::AVM) = mean(f.μ, X)
cov(f::GP, X::AVM) = cov(f.k, X)
xcov(f::GP, X::AVM, X′::AVM) = xcov(f.k, X, X′)
xcov(f::GP, f′::GP, X::AVM, X′::AVM) = xcov(kernel(f, f′), X, X′)

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
    rand(rng::AbstractRNG, f::GP, X::AM, N::Int=1)

Obtain `N` independent samples from the GP `f` at `X` using `rng`.
"""
rand(rng::AbstractRNG, f::GP, X::AVM, N::Int) =
    mean(f, X) .+ chol(cov(f, X))' * randn(rng, size(X, 1), N)
rand(rng::AbstractRNG, f::GP, X::AVM) = vec(rand(rng, f, X, 1))
