import Base: size, ==, +, -, *, isapprox, getindex, IndexStyle, map, broadcast,
    cov, logdet, chol, \, Matrix, UpperTriangular
export cov, LazyPDMat, Xt_A_X, Xt_A_Y, Xt_invA_Y, Xt_invA_X

"""
    logdet(U::UpperTriangular)

Compute the log determinant by summing the logdet of the diagonal of an `UpperTriangular`
matrix. Implementing this here is a horrible bit of type piracy, but it's necessary.
"""
logdet(U::UpperTriangular) = sum(logdet, view(U, diagind(U)))

"""
    LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}

A positive definite matrix which evaluates its Cholesky lazily and caches the result.
Please don't mutate it this object: `setindex!` isn't defined for a reason.
"""
mutable struct LazyPDMat{T<:Real, TΣ<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Σ::TΣ
    U::Union{Void, UpperTriangular{T}}
    ϵ::T
    LazyPDMat(Σ::TΣ) where {T<:Real, TΣ<:AM{T}} = new{T, TΣ}(Σ, nothing, 1e-6)
    LazyPDMat(Σ::TΣ, ϵ::Real) where TΣ<:AM{T} where T<:Real = new{T, TΣ}(Σ, nothing, ϵ)
end
LazyPDMat(Σ::LazyPDMat) = Σ
LazyPDMat(σ::Real) = σ
@inline unbox(Σ::LazyPDMat) = Σ.Σ

size(Σ::LazyPDMat) = size(unbox(Σ))
@inline getindex(Σ::LazyPDMat, i::Int...) = getindex(unbox(Σ), i...)
IndexStyle(::Type{<:LazyPDMat}) = IndexLinear()
==(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) == unbox(Σ2)
isapprox(Σ1::LazyPDMat, Σ2::LazyPDMat) = isapprox(unbox(Σ1), unbox(Σ2))

# Unary functions.
logdet(Σ::LazyPDMat) = 2 * logdet(chol(Σ))
function chol(Σ::LazyPDMat)
    if Σ.U == nothing
        Σ.U = chol(Symmetric(unbox(Σ) + Σ.ϵ * I))
    end
    return Σ.U
end

# Binary functions.
+(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) + unbox(Σ2))
+(Σ1::LazyPDMat, Σ2::UniformScaling) = (Σ2.λ > 0 ? LazyPDMat : identity)(unbox(Σ1) + Σ2)
-(Σ1::LazyPDMat, Σ2::LazyPDMat) = unbox(Σ1) - unbox(Σ2)
*(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) * unbox(Σ2))
map(::typeof(+), Σs::LazyPDMat...) = LazyPDMat(map(+, map(unbox, Σs)...))
map(::typeof(*), Σs::LazyPDMat...) = LazyPDMat(map(*, map(unbox, Σs)...))

broadcast(::typeof(*), Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(unbox(Σ1) .* unbox(Σ2))

# Specialised operations to exploit the Cholesky.
Xt_A_X(A::LazyPDMat, X::AbstractMatrix) = LazyPDMat(Symmetric(X' * unbox(A) * X))
Xt_A_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A) * x)
Xt_A_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A) * X)' * (chol(A) * Y)
function Xt_invA_X(A::LazyPDMat, X::AVM)
    V = chol(A)' \ X
    return LazyPDMat(Symmetric(V'V))
end
Xt_invA_X(A::LazyPDMat, x::AbstractVector) = sum(abs2, chol(A)' \ x)
Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A)' \ X)' * (chol(A)' \ Y)
\(Σ::LazyPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)

# Some generic operations that are useful for operations involving covariance matrices.
diag_AᵀA(A::AbstractMatrix) = vec(sum(abs2, A, 1))
function diag_AᵀB(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    return vec(sum(A .* B, 1))
end

diag_Xᵀ_A_X(A::LazyPDMat, X::AbstractMatrix) = diag_AᵀA(chol(A) * X)
diag_Xᵀ_A_Y(X::AM, A::LazyPDMat, Y::AM) = diag_AᵀB(chol(A) * X, chol(A) * Y)

diag_Xᵀ_invA_X(A::LazyPDMat, X::AbstractMatrix) = diag_AᵀA(chol(A)' \ X)
diag_Xᵀ_invA_Y(X::AM, A::LazyPDMat, Y::AM) = diag_AᵀB(chol(A)' \ X, chol(A)' \ Y)

# Specialised solver routine, especially useful for Titsias conditionals.
function Xtinv_A_Xinv(A::LazyPDMat, X::LazyPDMat)
    @assert size(A) == size(X)
    C = ((chol(A) / chol(X)) / chol(X)')
    return LazyPDMat(Symmetric(C'C))
end
