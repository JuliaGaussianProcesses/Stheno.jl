import PDMats: AbstractPDMat, invquad, dim

import Base: size, ==, full
import LinearAlgebra: cov, logdet, chol
import LinearAlgebra.BLAS: trsv
export cov, cov!, invquad, AbstractPDMat, Xt_invA_Y

const __ϵ = 1e-12

"""
    StridedPDMatrix

A strided positive definite matrix. This is mutable, but please don't mutate it.
"""
mutable struct StridedPDMatrix{T<:Real} <: AbstractPDMat{T}
    Σ::Symmetric{T}
    U::Union{Nothing, UpperTriangular{T}}
    StridedPDMatrix(Σ::Symmetric{T}) where T = new{T}(Σ, nothing)
end
StridedPDMatrix(Σ::AbstractMatrix) = StridedPDMatrix(Symmetric(Σ))
dim(Σ::StridedPDMatrix) = size(Σ.Σ, 1)
full(Σ::StridedPDMatrix) = Σ.Σ
logdet(Σ::StridedPDMatrix) = 2 * sum(log, view(chol(Σ), diagind(chol(Σ))))
invquad(Σ::StridedPDMatrix, x::AbstractVector) = sum(abs2, trsv('U', 'T', 'N', chol(Σ).data, x))
Xt_invA_Y(X::AVM, A::AbstractPDMat, Y::AVM) = (X' / chol(A)) * (chol(A)' \ Y)
function chol(Σ::StridedPDMatrix)
    if Σ.U == nothing
        Σ.U = chol(Σ.Σ + __ϵ * I)
    end
    return Σ.U
end
==(Σ1::StridedPDMatrix, Σ2::StridedPDMatrix) = Σ1.Σ == Σ2.Σ
