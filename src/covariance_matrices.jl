import Base.BLAS: trsv
import PDMats: AbstractPDMat, invquad, dim
import Base: cov, logdet, full, size, chol, ==
export cov, invquad, AbstractPDMat

"""
    StridedPDMatrix

A strided positive definite matrix, represented in terms of it's Cholesky factorization `U`,
which should be an upper triangular matrix.
"""
struct StridedPDMatrix{T<:Real} <: AbstractPDMat{T}
    U::UpperTriangular{T}
end

dim(Σ::StridedPDMatrix) = size(Σ.U, 1)
full(Σ::StridedPDMatrix) = Σ.U'Σ.U
logdet(Σ::StridedPDMatrix) = 2 * sum(log, view(Σ.U, diagind(Σ.U)))
invquad(Σ::StridedPDMatrix, x::AbstractVector) = sum(abs2, trsv('U', 'T', 'N', Σ.U.data, x))
chol(Σ::StridedPDMatrix) = Σ.U
==(Σ1::StridedPDMatrix, Σ2::StridedPDMatrix) = Σ1.U == Σ2.U

"""
    cov(k::Kernel, x::T, y::T) where T<:Union{AbstractVector, RowVector}

Allocate memory for the covariance matrix and call `cov!`.
"""
cov(k::Kernel, x::AbstractVector) = StridedPDMatrix(chol(Symmetric(k.(x, RowVector(x)))))
cov(k::Kernel, x::RowVector) = StridedPDMatrix(chol(Symmetric(k.(x.vec, x))))
cov(k::Kernel, x::T, y::T) where T<:RowVector = k.(x.vec, y)
cov(k::Kernel, x::T, y::T) where T<:AbstractVector = k.(x, RowVector(y))


"""
    

Compute the covariance matrix implied by a collection of 
"""
function cov(obs::Vararg{Tuple{GP, T, T} where T<:Union{AbstractVector, RowVector}})
    
end
