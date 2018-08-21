using LinearAlgebra, Statistics

import Statistics: cov
import LinearAlgebra: logdet, AbstractTriangular, Adjoint, transpose, adjoint,
    AbstractMatrix, cholesky, Cholesky
import Base: size, getindex, IndexStyle, ==, isapprox, +, -, *, map, broadcast, \

"""
    logdet(U::UpperTriangular)

Compute the log determinant by summing the logdet of the diagonal of an `UpperTriangular`
matrix. Implementing this here is a horrible bit of type piracy, but it's necessary.
"""
logdet(T::AbstractTriangular) = sum(logdet, view(T, diagind(T)))
logdet(A::Adjoint{T, <:AbstractTriangular{T}}) where T = logdet(A')

"""
    LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}

A positive definite matrix which evaluates its Cholesky lazily and caches the result.
Please don't mutate it this object: `setindex!` isn't defined for a reason.
"""
mutable struct LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Σ::TΣ
    C::Union{Nothing, Cholesky}
    ϵ::T
    LazyPDMat(Σ::TΣ) where {T<:Real, TΣ<:AbstractMatrix{T}} = new{T, TΣ}(Σ, nothing, 1e-6)
    function LazyPDMat(Σ::TΣ, ϵ::Real) where {T<:Real, TΣ<:AbstractMatrix{T}}
        return new{T, TΣ}(Σ, nothing, ϵ)
    end
end
@inline LazyPDMat(Σ::LazyPDMat) = Σ
@inline LazyPDMat(σ::Real) = σ
@inline unbox(Σ::LazyPDMat) = Σ.Σ
@inline unbox(A) = A
@inline AbstractMatrix(Σ::LazyPDMat) = unbox(Σ)

size(Σ::LazyPDMat) = size(unbox(Σ))
@inline getindex(Σ::LazyPDMat, i::Int...) = getindex(unbox(Σ), i...)
IndexStyle(::Type{<:LazyPDMat}) = IndexLinear()
==(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) == unbox(Σ2)
isapprox(Σ1::LazyPDMat, Σ2::LazyPDMat) = isapprox(unbox(Σ1), unbox(Σ2))

# Unary functions.
logdet(Σ::LazyPDMat) = logdet(cholesky(Σ))
function cholesky(Σ::LazyPDMat{T, <:AbstractMatrix{T}} where T<:Real)
    if Σ.C == nothing
        Σ.C = cholesky(Symmetric(unbox(Σ) + Σ.ϵ * I))
    end
    return Σ.C
end
chol(Σ) = cholesky(Σ).U

# Binary functions.
+(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) + unbox(Σ2))
+(Σ1::LazyPDMat, Σ2::UniformScaling) = (Σ2.λ > 0 ? LazyPDMat : identity)(unbox(Σ1) + Σ2)
-(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) - unbox(Σ2)
*(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) * unbox(Σ2))
map(::typeof(+), Σs::LazyPDMat...) = LazyPDMat(map(+, map(unbox, Σs)...))
map(::typeof(*), Σs::LazyPDMat...) = LazyPDMat(map(*, map(unbox, Σs)...))

broadcast(::typeof(*), Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) .* unbox(Σ2))

# Specialised operations to exploit the Cholesky.
@noinline function Xt_A_X(A::LazyPDMat, X::AbstractMatrix)
    return LazyPDMat(Symmetric(X' * unbox(unbox(A)) * X))
end
@noinline Xt_A_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A) * x)
@noinline Xt_A_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A) * X)' * (chol(A) * Y)
@noinline function Xt_invA_X(A::LazyPDMat, X::AVM)
    V = chol(A)' \ X
    return LazyPDMat(Symmetric(V'V))
end
@noinline Xt_invA_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A)' \ x)
# @noinline Xt_invA_X(A::LazyPDMat{<:Real, <:SymmetricToeplitz}, x::AV) = sum(abs2, chol(A) \ x)
@noinline Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A)' \ X)' * (chol(A)' \ Y)
@noinline \(Σ::LazyPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)

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
