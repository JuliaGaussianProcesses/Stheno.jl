using LinearAlgebra, Statistics

import Statistics: cov
import LinearAlgebra: logdet, AbstractTriangular, Adjoint, transpose, adjoint,
    AbstractMatrix
import Base: size, getindex, IndexStyle, ==, isapprox, +, -, *, map, broadcast, \

"""
    LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}

A positive definite matrix which evaluates its Cholesky lazily and caches the result.
Please don't mutate it this object: `setindex!` isn't defined for a reason!
"""
mutable struct LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Σ::TΣ
    U::Union{Nothing, AbstractMatrix}
    ϵ::T
    LazyPDMat(Σ::TΣ) where {T<:Real, TΣ<:AbstractMatrix{T}} = new{T, TΣ}(Σ, nothing, 1e-6)
    function LazyPDMat(Σ::TΣ, ϵ::Real) where {T<:Real, TΣ<:AbstractMatrix{T}}
        return new{T, TΣ}(Σ, nothing, ϵ)
    end
end
LazyPDMat(Σ::LazyPDMat) = Σ
LazyPDMat(σ::Real) = σ
unbox(Σ::LazyPDMat) = Σ.Σ
unbox(A) = A
AbstractMatrix(Σ::LazyPDMat) = unbox(Σ)

size(Σ::LazyPDMat) = size(unbox(Σ))
getindex(Σ::LazyPDMat, i::Int...) = getindex(unbox(Σ), i...)
IndexStyle(::Type{<:LazyPDMat}) = IndexLinear()
==(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) == unbox(Σ2)
isapprox(Σ1::LazyPDMat, Σ2::LazyPDMat) = isapprox(unbox(Σ1), unbox(Σ2))

# Unary functions.
chol(Σ::AbstractMatrix) = chol(Symmetric(Σ))
chol(Σ::Symmetric) = cholesky(Σ).U
function chol(Σ::LazyPDMat)
    if Σ.U == nothing
        Σ.U = chol(Symmetric(Σ.Σ + Σ.ϵ * I))
    end
    return Σ.U
end
logdet(Σ::LazyPDMat) = 2 * logdet(chol(Σ))[1]

# Binary functions.
+(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) + unbox(Σ2))
+(Σ1::LazyPDMat, Σ2::UniformScaling) = (Σ2.λ > 0 ? LazyPDMat : identity)(unbox(Σ1) + Σ2)
-(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) - unbox(Σ2)
*(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) * unbox(Σ2))
map(::typeof(+), Σs::LazyPDMat...) = LazyPDMat(map(+, map(unbox, Σs)...))
map(::typeof(*), Σs::LazyPDMat...) = LazyPDMat(map(*, map(unbox, Σs)...))

broadcast(::typeof(*), Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) .* unbox(Σ2))

# Specialised operations to exploit the Cholesky.
Xt_A_X(A::LazyPDMat, X::AbstractMatrix) = LazyPDMat(Symmetric(X' * unbox(unbox(A)) * X))
Xt_A_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A) * x)
Xt_A_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A) * X)' * (chol(A) * Y)
function Xt_invA_X(A::LazyPDMat, X::AVM)
    V = chol(A)' \ X
    return LazyPDMat(Symmetric(V'V))
end
Xt_invA_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A)' \ x)
# Xt_invA_X(A::LazyPDMat{<:Real, <:SymmetricToeplitz}, x::AV) = sum(abs2, chol(A) \ x)
Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A)' \ X)' * (chol(A)' \ Y)
\(Σ::LazyPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)

# Some generic operations that are useful for operations involving covariance matrices.
diag_AᵀA(A::AbstractMatrix) = vec(sum(abs2, A, dims=1))
function diag_AᵀB(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    return vec(sum(A .* B, dims=1))
end

diag_Xᵀ_A_X(A::LazyPDMat, X::AbstractMatrix) = diag_AᵀA(chol(A) * X)
diag_Xᵀ_A_Y(X::AM, A::LazyPDMat, Y::AM) = diag_AᵀB(chol(A) * X, chol(A) * Y)

diag_Xᵀ_invA_X(A::LazyPDMat, X::AbstractMatrix) = diag_AᵀA(chol(A)' \ X)
diag_Xᵀ_invA_Y(X::AM, A::LazyPDMat, Y::AM) = diag_AᵀB(chol(A)' \ X, chol(A)' \ Y)

# Specialised solver routine, especially useful for Titsias conditionals.
function Xtinv_A_Xinv(A::LazyPDMat, X::LazyPDMat)
    @assert size(A) == size(X)
    C = chol(X) \ (chol(X)' \ chol(A)')
    return LazyPDMat(Symmetric(C * C'))
end

transpose(A::LazyPDMat) = A
adjoint(A::LazyPDMat) = A
