import Base: size, show, broadcast!
import Base.LinAlg: ldiv!
export Conditional

# Internal data structure used to cache various quantities to prevent recomputation.
struct ConditionalData
    U::UpperTriangular
    tmp::Vector{Float64}
    tmp′::Vector{Float64}
end
ConditionalData(U::UpperTriangular) =
    ConditionalData(
        U,
        Vector{Float64}(uninitialized, size(U, 1)),
        Vector{Float64}(uninitialized, size(U, 1)),
    )
==(a::ConditionalData, b::ConditionalData) = a.U == b.U

"""
    ConditionalMean{Tμ<:μFun, Tk<:Kernel} <: μFun

A mean function used in conditional distributions.
"""
struct ConditionalMean{Tμ<:μFun, Tk<:Vector{<:Kernel}, Tα<:Vector{<:Real}} <: μFun
    μ::Tμ
    k_ff′::Tk
    α::Tα
end
ConditionalMean(μ::μFun, k_ff′::Vector{<:Kernel}, δ::Vector, data::ConditionalData) =
    ConditionalMean(μ, k_ff′, ldiv!(data.U, ldiv!(Transpose(data.U), δ)))
function (μ::ConditionalMean)(x::Real)
    kfs = [k isa LhsFinite ? Finite(k, [x]) : Finite(k.k, k.x, [k.y[x]]) for k in μ.k_ff′]
    return μ.μ(x) + dot(reshape(cov(reshape(kfs, :, 1)), :), μ.α)
end

"""
    Conditional{Tk} <: Kernel{NonStationary}

A kernel for use in conditional distributions.
"""
struct Conditional{Tk<:Kernel} <: Kernel{NonStationary}
    k_ff′::Tk
    k_f̂f::Vector{<:Kernel}
    k_f̂f′::Vector{<:Kernel}
    data::ConditionalData
end
Conditional(k_ff′::Kernel, k_f̂f::Kernel, k_f̂f′::Kernel, data::ConditionalData) = 
    Conditional(k_ff′, Vector{Kernel}([k_f̂f]), Vector{Kernel}([k_f̂f]), data)
Conditional(k_ff′::Kernel, k_f̂f::Kernel, k_f̂f′::Vector{<:Kernel}, data::ConditionalData) =
    Conditional(k_ff′, Vector{Kernel}([k_f̂f]), k_f̂f, data)
Conditional(k_ff′::Kernel, k_f̂f::Vector{<:Kernel}, k_f̂f′::Kernel, data::ConditionalData) =
    Conditional(k_ff′, k_f̂f, Vector{Kernel}([k_f̂f]), data)
function (k::Conditional)(x::Real, x′::Real)
    kfs = [k isa LhsFinite ? Finite(k, [x]) : Finite(k.k, k.x, [k.y[x]]) for k in k.k_f̂f]
    kf′s = [k isa LhsFinite ? Finite(k, [x′]) : Finite(k.k, k.x, [k.y[x′]]) for k in k.k_f̂f′]
    Ut = Transpose(k.data.U)
    a = ldiv!(Ut, cov(reshape(kfs, :, 1)))
    b = ldiv!(Ut, cov(reshape(kf′s, :, 1)))
    return k.k_ff′(x, x′) - (Transpose(a) * b)[1, 1]
end
function broadcast!(
    k::Conditional,
    K::AbstractMatrix,
    x::UnitRange{Int},
    x′::RowVector{Int, UnitRange{Int}},
)
    kfs = [k isa LhsFinite ? Finite(k, x) : Finite(k.k, k.x, k.y[x]) for k in k.k_f̂f]
    kf′s = [k isa LhsFinite ? Finite(k, x′') : Finite(k.k, k.x, k.y[x′']) for k in k.k_f̂f′]
    Ut = Tranpose(k.data.U)
    a = ldiv!(Ut, cov(reshape(kfs, :, 1)))
    b = ldiv!(Ut, cov(reshape(kf′s, :, 1)))
    return BLAS.gemm!('T', 'N', -1.0, a, b, 1.0, cov!(K, Finite(k.k_ff′, x, x′)))
end

==(a::Conditional{Tk}, b::Conditional{Tk}) where Tk =
    a.k_ff′ == b.k_ff′ && a.k_ff̂ == b.k_ff̂ && a.k_f̂f′ == b.k_f̂f′ && a.data == b.data
dims(k::Conditional) = dims(k.k_ff′)
size(k::Conditional) = size(k.k_ff′)
size(k::Conditional, n::Int) = size(k.k_ff′, n)
show(io::IO, k::Conditional) = print(io, "Conditional with prior kernel $(k.k_ff′)")
isfinite(k::Conditional) = isfinite(k.k_ff′)
