export ITMean, ITKernel, pick_dims, periodic

"""
    ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction

"InputTransformationMean": A `MeanFunction` `μ_it` defined by applying the function `f` to
the inputs to another `MeanFunction` `μ`. Concretely, `mean(μ_it, X) = mean(μ, f(X))`.
"""
struct ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction
    μ::Tμ
    f::Tf
end
ITMean(μ::MeanFunction, ::typeof(identity)) = μ
length(μ::ITMean) = length(μ.μ)
mean(μ::ITMean, X::AVM) = mean(μ.μ, μ.f(X))

"""
    ITKernel{Tk<:Kernel, Tf} <: Kernel

"InputTransformationKernel": An `ITKernel` `kit` is the kernel defined by applying a
transform `f` to the argument to a kernel `k`. Concretely:
`xcov(kit, X, X′) = xcov(k, f(X), f(X′))`, and by analogy `cov(kit, X) = cov(k, f(X))`.
"""
struct ITKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end
ITKernel(k::Kernel, ::typeof(identity)) = k
size(k::ITKernel, N::Int...) = size(k.k, N...)
isstationary(k::ITKernel) = isstationary(k.k)
cov(k::ITKernel, X::AVM) = cov(k.k, k.f(X))
marginal_cov(k::ITKernel, X::AVM) = marginal_cov(k.k, k.f(X))
xcov(k::ITKernel, X::AVM, X′::AVM) = xcov(k.k, k.f(X), k.f(X′))

"""
    pick_dims(x::Union{MeanFunction, Kernel}, I)

Returns either an `ITMean` or `ITKernel` which uses the columns of the input matrix `X`
specified by `I`.
"""
pick_dims(μ::MeanFunction, I) = ITMean(μ, X::AVM->X[:, I])
pick_dims(k::Kernel, I) = ITKernel(k, X::AVM->X[:, I])

"""
    periodic(k::Kernel, θ::Real)

Make `k` periodic with period `f`.
"""
periodic(k::Kernel, f::Real) =
    ITKernel(k, t::AV->hcat(cos.((2π * f) .* t), sin.((2π * f) .* t)))
