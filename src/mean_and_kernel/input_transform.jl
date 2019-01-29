"""
    ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction

"InputTransformationMean": A `MeanFunction` `μ_it` defined by applying the function `f` to
the inputs to another `MeanFunction` `μ`. Concretely, `mean(μ_it, X) = mean(μ, f(X))`.
"""
struct ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction
    μ::Tμ
    f::Tf
end
_map(μ::ITMean, X::AV) = _map(μ.μ, μ.f.(X))


"""
    ITKernel{Tk<:Kernel, Tf} <: Kernel

"InputTransformationKernel": An `ITKernel` `kit` is the kernel defined by applying a
transform `f` to the argument to a kernel `k`. Concretely: `k(x, x′) = k(f(x), f(x′))`.
"""
struct ITKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end

# Binary methods.
_map(k::ITKernel, x::AV, x′::AV) = _map(k.k, k.f.(x), k.f.(x′))
_pw(k::ITKernel, x::AV, x′::AV) = _pw(k.k, k.f.(x), k.f.(x′))

# Unary methods.
_map(k::ITKernel, x::AV) = _map(k.k, k.f.(x))
_pw(k::ITKernel, x::AV) = _pw(k.k, k.f.(x))


"""
    LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"LhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(f(x), x′)`.
"""
struct LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
_map(k::LhsITCross, x::AV, x′::AV) = _map(k.k, k.f.(x), x′)
_pw(k::LhsITCross, x::AV, x′::AV) = _pw(k.k, k.f.(x), x′)


"""
    RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"RhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(x, f(x′))`.
"""
struct RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
_map(k::RhsITCross, x::AV, x′::AV) = _map(k.k, x, k.f.(x′))
_pw(k::RhsITCross, x::AV, x′::AV) = _pw(k.k, x, k.f.(x′))


"""
    ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel

"InputTransformationCrossKernel": the kernel `k′` given by `k′(x, x′) = k(f(x), f′(x′))`.
"""
struct ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel
    k::Tk
    f::Tf
    f′::Tf′
end
_map(k::ITCross, x::AV, x′::AV) = _map(k.k, k.f.(x), k.f′.(x′))
_pw(k::ITCross, x::AV, x′::AV) = _pw(k.k, k.f.(x), k.f′.(x′))


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
    Scale{T<:Real}

Scale all elements of the inputs by `l`.
"""
struct Scale{T<:Real}
    l::T
end
(s::Scale)(x) = s.l * x
broadcasted(s::Scale, x::StepRangeLen) = s.l .* x
broadcasted(s::Scale, x::ColsAreObs) = ColsAreObs(s.l .* x.X)


"""
    LinearTransform{T<:AbstractMatrix}

A linear transformation of the inputs.
"""
struct LinearTransform{T<:AbstractMatrix}
    A::T
end
(l::LinearTransform)(x::AbstractVector) = l.A * x
broadcasted(l::LinearTransform, x::ColsAreObs) = ColsAreObs(l.A * x.X)


"""
    PickDims{Tidx}

Use inside an input transformation meanfunction / crosskernel to improve peformance.
"""
struct PickDims{Tidx}
    idx::Tidx
end
(f::PickDims)(x) = x[f.idx]
broadcasted(f::PickDims, x::ColsAreObs) = ColsAreObs(x.X[f.idx, :])
broadcasted(f::PickDims{<:Integer}, x::ColsAreObs) = x.X[f.idx, :]


"""
    Periodic{Tf<:Real}

Make a kernel or mean function periodic by projecting into two dimensions.
"""
struct Periodic{Tf<:Real}
    f::Tf
end
(p::Periodic)(t::Real) = [cos((2π * p.f) * t), sin((2π * p.f) * t)]
function broadcasted(p::Periodic, x::AbstractVector{<:Real})
    return ColsAreObs(vcat(cos.((2π * p.f) .* x)', sin.((2π * p.f) .* x)'))
end
