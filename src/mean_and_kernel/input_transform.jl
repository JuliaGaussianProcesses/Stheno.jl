import Base: map, eachindex
import Distances: pairwise
export transform, pick_dims, periodic, scale, Indexer

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
eachindex(μ::ITMean) = eachindex(μ.μ)
_map(μ::ITMean, X::AV) = map(μ.μ, map(μ.f, X))

"""
    ITKernel{Tk<:Kernel, Tf} <: Kernel

"InputTransformationKernel": An `ITKernel` `kit` is the kernel defined by applying a
transform `f` to the argument to a kernel `k`. Concretely: `k(x, x′) = k(f(x), f(x′))`.
"""
struct ITKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end
(k::ITKernel)(x) = k.k(k.f(x))
(k::ITKernel)(x, x′) = k.k(k.f(x), k.f(x′))
ITKernel(k::Kernel, ::typeof(identity)) = k
length(k::ITKernel) = length(k.k)
isstationary(k::ITKernel) = isstationary(k.k)
eachindex(k::ITKernel) = eachindex(k.k)

"""
    LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"LhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(f(x), x′)`.
"""
struct LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
(k::LhsITCross)(x, x′) = k.k(k.f(x), x′)
size(k::LhsITCross, n::Int) = size(k.k, n)

"""
    RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"RhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(x, f(x′))`.
"""
struct RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
(k::RhsITCross)(x, x′) = k.k(x, k.f(x′))
size(k::RhsITCross, n::Int) = size(k.k, n)

"""
    ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel

"InputTransformationCrossKernel": the kernel `k′` given by `k′(x, x′) = k(f(x), f′(x′))`.
"""
struct ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel
    k::Tk
    f::Tf
    f′::Tf′
end
(k::ITCross)(x, x′) = k.k(k.f(x), k.f′(x′))
size(k::ITCross, n::Int) = size(k.k, n)

_map(k::ITKernel, X::AV) = map(k.k, map(k.f, X))
_map(k::ITKernel, X::AV, X′::AV) = map(k.k, map(k.f, X), map(k.f, X′))
_pairwise(k::ITKernel, X::AV) = pairwise(k.k, map(k.f, X))
_pairwise(k::ITKernel, X::AV, X′::AV) = pairwise(k.k, map(k.f, X), map(k.f, X′))

"""
    transform(f::Union{MeanFunction, Kernel}, ϕ)

Applies the input-transform `ϕ` to `f`.
"""
transform(μ::MeanFunction, ϕ) = ITMean(μ, ϕ)

transform(μ::MeanFunction, ::typeof(identity)) = μ
transform(μ::ZeroMean, ϕ) = μ
transform(μ::ZeroMean, ::typeof(identity)) = μ

transform(k::Kernel, ϕ) = ITKernel(k, ϕ)
transform(k::Kernel, ::typeof(identity)) = k
transform(k::ZeroKernel, ϕ) = k
transform(k::ZeroKernel, ::typeof(identity)) = k

transform(k::CrossKernel, ϕ, ::Val{1}) = LhsITCross(k, ϕ)
transform(k::CrossKernel, ϕ, ::Val{2}) = RhsITCross(k, ϕ)

transform(k::ZeroKernel, ϕ, ::Val{1}) = k
transform(k::ZeroKernel, ϕ, ::Val{2}) = k

transform(k::CrossKernel, ϕ, ϕ′) = ITCross(k, ϕ, ϕ′)


"""
    scale(f::Union{MeanFunction, Kernel}, l::Real)

Multiply each element of the input by `l`.
"""
scale(f::Union{MeanFunction, Kernel}, l::Real) = transform(f, Scale(l))

struct Scale{T<:Real}
    l::T
end
(s::Scale)(x) = s.l * x
map(s::Scale, x::AVM) = s.l .* x

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
periodic(k::EQ, p::Real) = PerEQ(p)
periodic(f::Union{MeanFunction, Kernel}) = periodic(f, 1.0)

struct Periodic{Tf<:Real}
    f::Tf
end
(p::Periodic)(t::Real) = [cos((2π * p.f) * t), sin((2π * p.f) * t)]
function map(p::Periodic, t::AV)
    return ColsAreObs(vcat(
        map(x->cos((2π * p.f) * x), t)',
        map(x->sin((2π * p.f) * x), t)',
    ))
end

"""
    Indexer

Pulls out the `n`th index of whatever object is passed.
"""
struct Indexer
    n::Int
end
(indexer::Indexer)(x) = x[indexer.n]
map(indexer::Indexer, x::ColsAreObs) = view(x.X, indexer.n, :)
