"""
    ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction

"InputTransformationMean": A `MeanFunction` `μ_it` defined by applying the function `f` to
the inputs to another `MeanFunction` `μ`. Concretely, `mean(μ_it, X) = mean(μ, f(X))`.
"""
struct ITMean{Tμ<:MeanFunction, Tf} <: MeanFunction
    μ::Tμ
    f::Tf
end
ew(μ::ITMean, X::AV) = ew(μ.μ, μ.f.(X))


"""
    ITKernel{Tk<:Kernel, Tf} <: Kernel

"InputTransformationKernel": An `ITKernel` `kit` is the kernel defined by applying a
transform `f` to the argument to a kernel `k`. Concretely: `kit(x, x′) = k(f(x), f(x′))`.
"""
struct ITKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end

# Binary methods.
ew(k::ITKernel, x::AV, x′::AV) = ew(k.k, k.f.(x), k.f.(x′))
pw(k::ITKernel, x::AV, x′::AV) = pw(k.k, k.f.(x), k.f.(x′))

# Unary methods.
ew(k::ITKernel, x::AV) = ew(k.k, k.f.(x))
pw(k::ITKernel, x::AV) = pw(k.k, k.f.(x))

"""
    LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"LhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(f(x), x′)`.
"""
struct LhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
ew(k::LhsITCross, x::AV, x′::AV) = ew(k.k, k.f.(x), x′)
pw(k::LhsITCross, x::AV, x′::AV) = pw(k.k, k.f.(x), x′)


"""
    RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel

"RhsInputTransformationCrossKernel": The kernel `k′` given by `k′(x, x′) = k(x, f(x′))`.
"""
struct RhsITCross{Tk<:CrossKernel, Tf} <: CrossKernel
    k::Tk
    f::Tf
end
ew(k::RhsITCross, x::AV, x′::AV) = ew(k.k, x, k.f.(x′))
pw(k::RhsITCross, x::AV, x′::AV) = pw(k.k, x, k.f.(x′))


"""
    ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel

"InputTransformationCrossKernel": the kernel `k′` given by `k′(x, x′) = k(f(x), f′(x′))`.
"""
struct ITCross{Tk<:Kernel, Tf, Tf′} <: CrossKernel
    k::Tk
    f::Tf
    f′::Tf′
end
ew(k::ITCross, x::AV, x′::AV) = ew(k.k, k.f.(x), k.f′.(x′))
pw(k::ITCross, x::AV, x′::AV) = pw(k.k, k.f.(x), k.f′.(x′))


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
    Stretch{T<:Real}

Stretch all elements of the inputs by `l`.
"""
struct Stretch{T<:Real}
    l::T
end
(s::Stretch)(x) = s.l * x
broadcasted(s::Stretch, x::StepRangeLen) = s.l .* x
broadcasted(s::Stretch, x::ColsAreObs) = ColsAreObs(s.l .* x.X)

stretch(f::Union{MeanFunction, Kernel, CrossKernel}, l::Real) = transform(f, Stretch(l))


"""
    LinearTransform{T<:AbstractMatrix}

A linear transformation of the inputs.
"""
struct LinearTransform{T<:AbstractMatrix}
    A::T
end
(l::LinearTransform)(x::AbstractVector) = l.A * x
broadcasted(l::LinearTransform, x::ColsAreObs) = ColsAreObs(l.A * x.X)

function stretch(f::Union{MeanFunction, Kernel, CrossKernel}, A::AbstractMatrix)
    return transform(f, LinearTransform(A))
end
function stretch(f::Union{MeanFunction, Kernel, CrossKernel}, A::AbstractVector)
    return stretch(f, Diagonal(A))
end


"""
    Select{Tidx}

Use inside an input transformation meanfunction / crosskernel to improve peformance.
"""
struct Select{Tidx}
    idx::Tidx
end
(f::Select)(x) = x[f.idx]
broadcasted(f::Select, x::ColsAreObs) = ColsAreObs(x.X[f.idx, :])
broadcasted(f::Select{<:Integer}, x::ColsAreObs) = x.X[f.idx, :]

function broadcasted(f::Select, x::AbstractVector{<:CartesianIndex})
    out = Matrix{Int}(undef, length(f.idx), length(x))
    for i in f.idx, n in eachindex(x)
        out[i, n] = x[n][i]
    end
    return ColsAreObs(out)
end
@adjoint function broadcasted(f::Select, x::AV{<:CartesianIndex})
    return broadcasted(f, x), Δ->(nothing, nothing)
end
function broadcasted(f::Select{<:Integer}, x::AV{<:CartesianIndex})
    out = Matrix{Int}(undef, length(x))
    for n in eachindex(x)
        out[n] = x[n][f.idx]
    end
    return out
end

select(f::Union{MeanFunction, Kernel, CrossKernel}, idx) = transform(f, Select(idx))


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

"""
    periodic(g::Union{MeanFunction, Kernel, CrossKernel}, f::Real)

Make `g` periodic with frequency `f`.
"""
periodic(g::Union{MeanFunction, Kernel, CrossKernel}, f::Real) = transform(g, Periodic(f))
