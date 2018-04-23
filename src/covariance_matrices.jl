import Base: size, ==, +, -, *, isapprox, getindex, IndexStyle, map, broadcast
import LinearAlgebra: cov, logdet, chol, \, Matrix, UpperTriangular
export cov, LazyPDMat, Xt_A_X, Xt_A_Y, Xt_invA_Y, Xt_invA_X

const __ϵ = 1e-9

# Define `logdet` sensibly for `UpperTriangular` matrices.
LinearAlgebra.logdet(U::UpperTriangular) = sum(LinearAlgebra.logdet, view(U, diagind(U)))

"""
    LazyPDMat{T<:Real} <: AbstractMatrix{T}

A positive definite matrix which evaluates its Cholesky lazily and caches the result.
Please don't mutate it this object: `setindex!` isn't defined for a reason.
"""
mutable struct LazyPDMat{T<:Real} <: AbstractMatrix{T}
    Σ::AbstractMatrix{T}
    U::Union{Nothing, UpperTriangular{T}}
    LazyPDMat(Σ::AbstractMatrix{T}) where T = new{T}(Σ, nothing)
end
LazyPDMat(Σ::LazyPDMat) = Σ
LazyPDMat(σ::Real) = σ
Matrix(Σ::LazyPDMat) = Matrix(Σ.Σ)
size(Σ::LazyPDMat) = size(Σ.Σ)
@inline getindex(Σ::LazyPDMat, i::Int...) = getindex(Σ.Σ, i...)
IndexStyle(::Type{<:LazyPDMat}) = IndexLinear()
==(Σ1::LazyPDMat, Σ2::LazyPDMat) = Σ1.Σ == Σ2.Σ
isapprox(Σ1::LazyPDMat, Σ2::LazyPDMat) = isapprox(Σ1.Σ, Σ2.Σ)

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
+(Σ1::LazyPDMat, Σ2::UniformScaling) = (Σ2.λ > 0 ? LazyPDMat : identity)(Σ1.Σ + Σ2)
-(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Matrix(Σ1) - Matrix(Σ2))
*(Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Matrix(Σ1) * Matrix(Σ2))
map(::typeof(*), Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(map(*, Σ1.Σ, Σ2.Σ))
broadcast(::typeof(*), Σ1::LazyPDMat, Σ2::LazyPDMat) = LazyPDMat(Σ1.Σ .* Σ2.Σ)

# Specialised operations to exploit the Cholesky.
function Xt_A_X(A::LazyPDMat, X::AVM)
    V = chol(A) * X
    return LazyPDMat(V'V)
end
Xt_A_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A) * X)' * (chol(A) * Y)
function Xt_invA_X(A::LazyPDMat, X::AVM)
    V = chol(A)' \ X
    return LazyPDMat(V'V)
end
Xt_invA_Y(X::AVM, A::LazyPDMat, Y::AVM) = (chol(A)' \ X)' * (chol(A)' \ Y)
\(Σ::LazyPDMat, X::Union{AM, AV}) = chol(Σ) \ (chol(Σ)' \ X)
