import PDMats: AbstractPDMat, invquad, dim
# import Base: cov, logdet, full, size, chol, ==

import Base: size, ==, full
import LinearAlgebra: cov, logdet, chol
import LinearAlgebra.BLAS: trsv
export cov, cov!, invquad, AbstractPDMat

"""
    StridedPDMatrix

A strided positive definite matrix, represented in terms of it's Cholesky factorization `U`.
"""
struct StridedPDMatrix{T<:Real} <: AbstractPDMat{T}
    U::UpperTriangular{T}
end
dim(Σ::StridedPDMatrix) = size(Σ.U, 1)
full(Σ::StridedPDMatrix) = Transpose(Σ.U) * Σ.U
logdet(Σ::StridedPDMatrix) = 2 * sum(log, view(Σ.U, diagind(Σ.U)))
invquad(Σ::StridedPDMatrix, x::AbstractVector) = sum(abs2, trsv('U', 'T', 'N', Σ.U.data, x))
chol(Σ::StridedPDMatrix) = Σ.U
==(Σ1::StridedPDMatrix, Σ2::StridedPDMatrix) = Σ1.U == Σ2.U

"""
    cov!(K::AbstractMatrix, k::Union{Kernel, Matrix{Kernel}})

Store in `K` the covariance matrix implied by the finite kernel (matrix thereof) `k`.
"""
cov!(K::AbstractMatrix, k::Kernel) = broadcast!(k, K, 1:size(k, 1), (1:size(k, 2))')
function cov!(K::AbstractMatrix, k::Matrix)
    rs_, cs_ = size.(k[:, 1], 1), size.(k[1, :], 2)
    rs = Vector{Int}(undef, length(rs_) + 1)
    cs = Vector{Int}(undef, length(cs_) + 1)
    cumsum!(view(rs, 2:length(rs_) + 1), rs_)
    cumsum!(view(cs, 2:length(cs_) + 1), cs_)
    rs[1], cs[1] = 0, 0
    for I in CartesianIndices(k)
        cov!(view(K, rs[I[1]]+1:rs[I[1]+1], cs[I[2]]+1:cs[I[2]+1]), k[I[1], I[2]])
    end
    return K
end

"""
    cov(k::Union{Kernel, Matrix{Kernel}})

Compute the covariance matrix implied by the finite kernel (or matrix thereof) `k`.
"""
cov(k::Kernel) = cov!(Matrix{Float64}(undef, size(k, 1), size(k, 2)), k)
cov(k::Matrix) = cov!(Matrix{Float64}(undef, sum(size.(k[:, 1], 1)), sum(size.(k[1, :], 2))), k)
