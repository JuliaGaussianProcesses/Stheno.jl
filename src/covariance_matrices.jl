import PDMats: AbstractPDMat, invquad, dim, Xt_invA_X

import Base: size, ==, +, -, *
import LinearAlgebra: cov, logdet, chol, \, Matrix, UpperTriangular
export cov, invquad, LazyPDMat, Xt_invA_Y, Xt_invA_X

const __ϵ = 1e-12

# Define `logdet` sensibly for `UpperTriangular` matrices.
LinearAlgebra.logdet(U::UpperTriangular) = sum(LinearAlgebra.logdet, view(U, diagind(U)))

"""
    LazyPDMat{T<:Real} <: AbstractPDMat{T}

A `PDMat` which evaluates its Cholesky lazily and caches the result.
This is mutable, but please don't mutate it.
"""
mutable struct LazyPDMat{T<:Real} <: AbstractPDMat{T}
    Σ::AbstractMatrix{T}
    U::Union{Nothing, UpperTriangular{T}}
    LazyPDMat(Σ::AbstractMatrix{T}) where T = new{T}(Σ, nothing)
end
dim(Σ::LazyPDMat) = size(Σ.Σ, 1)
Matrix(Σ::LazyPDMat) = Matrix(Σ.Σ)
==(Σ1::LazyPDMat, Σ2::LazyPDMat) = Σ1.Σ == Σ2.Σ

# Unary functions.
LinearAlgebra.logdet(Σ::LazyPDMat) = 2 * LinearAlgebra.logdet(chol(Σ))
function LinearAlgebra.chol(Σ::LazyPDMat)
    if Σ.U == nothing
        Σ.U = chol(Σ.Σ + __ϵ * I)
    end
    return Σ.U
end

# Binary functions.
+(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Matrix(Σ1) + Matrix(Σ2))
-(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Matrix(Σ1) - Matrix(Σ2))
*(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Matrix(Σ1) * Matrix(Σ2))
invquad(Σ::LazyPDMat, x::AV) = sum(abs2, chol(Σ)' \ x)
function Xt_invA_X(A::LazyPDMat, X::AM)
    V = chol(A)' \ X
    return LazyPDMat(V'V)
end
Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A)' \ X)' * (chol(A)' \ Y)
# Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (X' / chol(A)) * (chol(A)' \ Y)
\(Σ::LazyPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)
