import Base: size, getindex, Matrix, *, \, /

"""
    WoodburyLazyPDMat{T} <: AbstractMatrix{T}

A lazily-represented matrix to which the Woodbury-Sherman-Morrison formula can be applied
for efficient operations. Is an `AbstractMatrix` `A`, given by `A := Xᵀ Σ X + σ^2 I`.
"""
struct WoodburyLazyPDMat{T<:Real, TX<:AM{T}, TΣ<:LazyPDMat{T}} <: AM{T}
    X::TX
    Σ::TΣ
    σ::T
end

function __intermediates(A::WoodburyLazyPDMat)
    Γ = chol(A.Σ) * A.X / A.σ
    Ω = Γ * Γ' + I
    return Γ, Ω
end

# AbstractArray interface.
size(A::WoodburyLazyPDMat) = (size(A.X, 2), size(A.X, 2))
function getindex(A::WoodburyLazyPDMat, p::Int, q::Int)
    return Xt_A_Y(view(A.X, :, p), A.Σ, view(A.X, :, q)) + (p == q) * A.σ^2
end

# Conversions.
LazyPDMat(A::WoodburyLazyPDMat) = Xt_A_X(A.Σ, A.X) + A.σ^2 * I
Matrix(A::WoodburyLazyPDMat) = Matrix(LazyPDMat(A))

# Efficient matrix multiplication.
function *(A::WoodburyLazyPDMat{<:Real}, B::VecOrMat{<:Real})
    return A.X' * (A.Σ * (A.X * B)) .+ (A.σ^2) .* B
end
function *(A::Matrix{<:Real}, B::WoodburyLazyPDMat{<:Real})
    return ((A * B.X') * B.Σ) * B.X .+ (B.σ^2) .* A
end

# Efficient linear system solving.
function \(A::WoodburyLazyPDMat, B::VecOrMat{<:Real})
    Γ, Ω = __intermediates(A)
    return (B .- Γ' * (Ω \ (Γ * B))) ./ A.σ^2
end
function /(B::VecOrMat{<:Real}, A::WoodburyLazyPDMat)
    Γ, Ω = __intermediates(A)
    return (B .- ((B * Γ') / Ω) * Γ) ./ A.σ^2
end
