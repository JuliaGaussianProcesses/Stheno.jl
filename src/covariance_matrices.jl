import PDMats: AbstractPDMat, invquad, dim, Xt_invA_X

import Base: size, ==, +, -, *
import LinearAlgebra: cov, logdet, chol, \, Matrix
export cov, invquad, AbstractPDMat, Xt_invA_Y, Xt_invA_X

const __ϵ = 1e-12

"""
    StridedPDMat

A strided positive definite matrix. This is mutable, but please don't mutate it.
"""
mutable struct StridedPDMat{T<:Real} <: AbstractPDMat{T}
    Σ::Symmetric{T}
    U::Union{Nothing, UpperTriangular{T}}
    StridedPDMat(Σ::Symmetric{T}) where T = new{T}(Σ, nothing)
end
StridedPDMat(Σ::AbstractMatrix) = StridedPDMat(Symmetric(Σ))
dim(Σ::StridedPDMat) = size(Σ.Σ, 1)
Symmetric(Σ::StridedPDMat) = Σ.Σ
Matrix(Σ::StridedPDMat) = Matrix(Σ.Σ)
==(Σ1::StridedPDMat, Σ2::StridedPDMat) = Σ1.Σ == Σ2.Σ

# Unary functions.
logdet(Σ::StridedPDMat) = 2 * sum(log, view(chol(Σ), diagind(chol(Σ))))
function chol(Σ::StridedPDMat)
    if Σ.U == nothing
        Σ.U = chol(Σ.Σ + __ϵ * I)
    end
    return Σ.U
end

# Binary functions.
+(Σ1::StridedPDMat, Σ2::StridedPDMat) = StridedPDMat(Matrix(Σ1) + Matrix(Σ2))
-(Σ1::StridedPDMat, Σ2::StridedPDMat) = StridedPDMat(Matrix(Σ1) - Matrix(Σ2))
*(Σ1::StridedPDMat, Σ2::StridedPDMat) = StridedPDMat(Matrix(Σ1) * Matrix(Σ2))
invquad(Σ::StridedPDMat, x::AV) = sum(abs2, chol(Σ)' \ x)
function Xt_invA_X(A::StridedPDMat, X::AM)
    V = chol(A)' \ X
    return StridedPDMat(Symmetric(V'V))
end
Xt_invA_Y(X::AVM, A::AbstractPDMat, Y::AVM) = (X' / chol(A)) * (chol(A)' \ Y)
\(Σ::StridedPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)
