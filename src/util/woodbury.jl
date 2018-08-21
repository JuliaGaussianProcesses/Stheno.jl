import Base: size, getindex, Matrix, *, \, /
import LinearAlgebra: logdet

"""
    WoodburyMat{T} <: AbstractMatrix{T}

A lazily-represented matrix to which the Woodbury-Sherman-Morrison formula can be applied
for efficient operations. Is an `AbstractMatrix` `A`, given by `A := Xᵀ Σ^{-1} X + σ^2 I`.
Note that this is a rather specific instantiation of the types of matrices considered by the
matrix inversion and determinant lemmas. Should probably be generalised at some point.
"""
struct WoodburyMat{T<:Real, TX<:AM{T}, TΣ<:LazyPDMat{T}} <: AM{T}
    X::TX
    Σ::TΣ
    σ::T
    Γ::TX
    Ω::TΣ
    function WoodburyMat(X::TX, Σ::TΣ, σ::T) where {T<:Real, TX<:AM{T}, TΣ<:LazyPDMat{T}}
        Γ = chol(Σ)' \ X ./ σ
        Ω = LazyPDMat(Γ * Γ' + I, 0)
        return new{T, TX, TΣ}(X, Σ, σ, Γ, Ω)
    end
end

# AbstractArray interface.
size(A::WoodburyMat) = (size(A.X, 2), size(A.X, 2))
function getindex(A::WoodburyMat, p::Int, q::Int)
    return Xt_invA_Y(view(A.X, :, p), A.Σ, view(A.X, :, q)) + (p == q) * A.σ^2
end

# Conversions.
LazyPDMat(A::WoodburyMat) = Xt_invA_X(A.Σ, A.X) + A.σ^2 * I
Matrix(A::WoodburyMat) = Matrix(LazyPDMat(A))

# Some unary operations.
logdet(A::WoodburyMat) = logdet(A.Ω) + 2 * size(A, 1) * log(A.σ)

# Efficient matrix multiplication.
*(A::WoodburyMat, B::VecOrMat) = A.X' * (A.Σ \ (A.X * B)) .+ (A.σ^2) .* B
*(A::Matrix, B::WoodburyMat) = ((A * B.X') / B.Σ) * B.X .+ (B.σ^2) .* A

# Efficient linear system solving.
\(A::WoodburyMat, B::VecOrMat) = (B .- A.Γ' * (A.Ω \ (A.Γ * B))) ./ A.σ^2
/(B::VecOrMat, A::WoodburyMat) = (B .- ((B * A.Γ') / A.Ω) * A.Γ) ./ A.σ^2

# Efficient inverse-quadratic-form-like computations.
Xt_invA_X(A::WoodburyMat, X::VecOrMat) = (X'X .- Xt_invA_X(A.Ω, A.Γ * X)) ./ (A.σ^2)
