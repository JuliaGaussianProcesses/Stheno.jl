export transform, pick_dims, periodic

"""
    ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction

"InputTransformationMean": A `MeanFunction` `μ_it` defined by applying the function `f` to
the inputs to another `MeanFunction` `μ`. Concretely, `mean(μ_it, X) = mean(μ, f(X))`.
"""
struct ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction
    μ::Tμ
    f::Tf
end
(μ::ITMean)(x) = μ.μ(μ.f(x))
ITMean(μ::MeanFunction, ::typeof(identity)) = μ
length(μ::ITMean) = length(μ.μ)
unary_obswise(μ::ITMean, X::AVM) = unary_obswise(μ.μ, unary_obswise(μ.f, X))

"""
    ITKernel{Tk<:Kernel, Tf} <: Kernel

"InputTransformationKernel": An `ITKernel` `kit` is the kernel defined by applying a
transform `f` to the argument to a kernel `k`. Concretely: `k(x, x′) = k(f(x), f(x′))`.
"""
struct ITKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end
(k::ITKernel)(x, x′) = k.k(k.f(x), k.f(x′))
ITKernel(k::Kernel, ::typeof(identity)) = k
size(k::ITKernel, N::Int...) = size(k.k, N...)
isstationary(k::ITKernel) = isstationary(k.k)

binary_obswise(k::ITKernel, X::AVM, X′::AVM) =
    binary_obswise(k.k, unary_obswise(k.f, X), unary_obswise(k.f, X′))
pairwise(k::ITKernel, X::AVM) = pairwise(k.k, unary_obswise(k.f, X))
pairwise(k::ITKernel, X::AVM, X′::AVM) =
    pairwise(k.k, unary_obswise(k.f, X), unary_obswise(k.f, X′))

"""
    transform(f::Union{MeanFunction, Kernel}, ϕ)

Applies the input-transform `ϕ` to `f`.
"""
transform(μ::MeanFunction, ϕ) = ITMean(μ, ϕ)
transform(k::Kernel, ϕ) = ITKernel(k, ϕ)

"""
    pick_dims(x::Union{MeanFunction, Kernel}, I)

Returns either an `ITMean` or `ITKernel` which uses the columns of the input matrix `X`
specified by `I`.
"""
pick_dims(μ::MeanFunction, I) = ITMean(μ, X::AV->X[I])
pick_dims(k::Kernel, I) = ITKernel(k, X::AV->X[I])

"""
    periodic(k::Union{MeanFunction, Kernel}, θ::Real)

Make `k` periodic with period `f`.
"""
periodic(μ::MeanFunction, f::Real) = ITMean(μ, Periodic(f))
periodic(k::Kernel, f::Real) = ITKernel(k, Periodic(f))

struct Periodic{Tf<:Real}
    f::Tf
end
(p::Periodic)(t::Real) = [cos((2π * p.f) * t), sin((2π * p.f) * t)]
unary_obswise(p::Periodic, t::AV) =
    vcat(RowVector(cos.((2π * p.f) .* t)),
         RowVector(sin.((2π * p.f) .* t)))
