# Finite covariance functions based on covariance matrices.

"""
    Finite <: Kernel

Kernel for a finite-dimensional GP. Fully specified by a matrix of values.
"""
abstract type Finite{T<:KernelType} <: Kernel{T} end

"""
    DenseFinite <: Finite{NonStationary}
"""
struct DenseFinite{T<:AbstractMatrix} <: Finite{NonStationary}
    Σ::AbstractMatrix
end
@inline (k::DenseFinite)(p::Int, q::Int) = k.Σ[p, q]
==(a::DenseFinite, b::DenseFinite) = a.Σ == b.Σ
DenseFinite(k::Kernel, x) = DenseFinite(cov(k, x))
