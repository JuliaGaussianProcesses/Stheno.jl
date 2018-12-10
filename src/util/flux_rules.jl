using Flux, Flux.Tracker
using Flux.Tracker: track, @grad, TrackedVecOrMat, data, tracker, Call

import Distances: pairwise
import Base: +
using LinearAlgebra: RealHermSymComplexHerm, copytri!
import LinearAlgebra: UpperTriangular, logdet, \, /, UpperTriangular, Symmetric, diag

# Binary pairwise (AbstractMatrix)
pairwise(s::SqEuclidean, X::TrackedMatrix, Y::TrackedMatrix) = track(pairwise, s, X, Y)
pairwise(s::SqEuclidean, X::TrackedMatrix, Y::AbstractMatrix) = track(pairwise, s, X, Y)
pairwise(s::SqEuclidean, X::AbstractMatrix, Y::TrackedMatrix) = track(pairwise, s, X, Y)
@grad function pairwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
    D = pairwise(s, data(x), data(y))
    return D, function(Δ)
        x̄ = 2 .* (x * Diagonal(sum(Δ; dims=2)) .- y * Δ')
        ȳ = 2 .* (y * Diagonal(sum(Δ; dims=1)) .- x * Δ)
        return nothing, x̄, ȳ
    end
end

# Unary pairwise (AbstractMatrix)
pairwise(s::SqEuclidean, X::TrackedMatrix) = track(pairwise, s, X)
@grad function pairwise(s::SqEuclidean, X::AbstractMatrix)
    D = pairwise(s, data(X))
    return D, Δ->(nothing, 4 * (X * (Diagonal(reshape(sum(Δ; dims=1), :)) - Δ)))
end

# Binary pairwise (AbstractVector)
pairwise(s::SqEuclidean, x::AbstractVector, y::AbstractVector) = (x .- y').^2
pairwise(s::SqEuclidean, x::TrackedVector, y::TrackedVector) = track(pairwise, s, x, y)
pairwise(s::SqEuclidean, x::TrackedVector, y::AbstractVector) = track(pairwise, s, x, y)
pairwise(s::SqEuclidean, x::AbstractVector, y::TrackedVector) = track(pairwise, s, x, y)
@grad function pairwise(s::SqEuclidean, x::AbstractVector, y::AbstractVector)
    D = pairwise(s, data(x), data(y))
    return D, function(Δ)
        x̄ = 2 .* (reshape(sum(Δ; dims=2), :) .* x .- Δ * y)
        ȳ = 2 .* (reshape(sum(Δ; dims=1), :)  .* y .- Δ' * x)
        return nothing, x̄, ȳ
    end
end

# Unary pairwise (AbstractVector)
pairwise(s::SqEuclidean, x::AbstractVector) = (x .- x').^2
pairwise(s::SqEuclidean, x::TrackedVector) = track(pairwise, s, x)
@grad function pairwise(s::SqEuclidean, x::AbstractVector)
    D = pairwise(s, data(x))
    return D, function(Δ)
        return nothing, 4 .* (reshape(sum(Δ; dims=1), :) .* x .- Δ * x)
    end
end

import Base: *
*(A::TrackedMatrix, B::Diagonal) = track(*, A, B)
*(A::Diagonal, B::TrackedMatrix) = track(*, A, B)

*(A::Adjoint{T, <:AbstractMatrix{T}} where T, b::TrackedVector) = track(*, A, b)

# Y = A \ B
\(A::TrackedMatrix, B::TrackedVecOrMat) = track(\, A, B)
\(A::AbstractMatrix, B::TrackedVecOrMat) = track(\, A, B)
\(A::TrackedMatrix, B::AbstractVecOrMat) = track(\, A, B)
\(A::Adjoint, B::TrackedMatrix) = track(\, A, B)
\(A::Adjoint{<:Real, <:UpperTriangular}, b::TrackedVector) = track(\, A, b)
@grad function \(A::AbstractMatrix, B::AbstractVecOrMat)
    Y = data(A) \ data(B)
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

# Y = A / B
/(A::TrackedMatrix, B::TrackedMatrix) = track(/, A, B)
/(A::AbstractMatrix, B::TrackedMatrix) = track(/, A, B)
/(A::TrackedMatrix, B::AbstractMatrix) = track(/, A, B)
/(A::Adjoint{T, <:AbstractVector{T}} where T, B::TrackedMatrix) = track(/, A, B)
@grad function /(A::AbstractMatrix, B::AbstractMatrix)
    Y = data(A) / data(B)
    return Y, function(Ȳ)
        Ā = Ȳ / B'
        return (Ā, -Y' * Ā)
    end
end

# Construction of Symmetric matrices.
Symmetric(A::TrackedMatrix) = track(Symmetric, A)
@grad function Symmetric(A::TrackedMatrix)
    return Symmetric(data(A)), Δ->(symmetric_back(Δ),)
end
symmetric_back(Δ) = UpperTriangular(Δ) + LowerTriangular(Δ)' - Diagonal(Δ)
symmetric_back(Δ::UpperTriangular) = Δ

# Just get the upper triangular bit of the cholesky decomposition.
chol(Σ::TrackedMatrix) = track(chol, Σ)
@grad function chol(Σ::AbstractMatrix{T}) where T
    U = chol(data(Σ))
    return U, function(Ū)
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(T), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] *= 0.5
        end
        return (UpperTriangular(Σ̄),)
    end
end

diag(A::TrackedMatrix) = track(diag, A)
@grad function diag(A::TrackedMatrix)
    return diag(data(A)), Δ->(Diagonal(Δ),)
end

# Specialised logdet sensitivity for UpperTriangular matrices.
const StridedTriangular{T} = AbstractTriangular{T, <:StridedMatrix{T}}
logdet(U::TrackedMatrix{T, <:StridedTriangular{T}} where T) = track(logdet, U)
@grad function logdet(U::TrackedMatrix{T, <:StridedTriangular{T}} where T)
    return logdet(data(U)), Δ->(UpperTriangular(Matrix(Diagonal(Δ ./ diag(data(U))))),)
end

# Addition of `TrackedMatrix` and `UniformScaling`, where it is assumed that the scaling
# will not be differentiated w.r.t.
+(A::TrackedMatrix, S::UniformScaling) = track(+, A, S)
@grad function +(A::TrackedMatrix, S::UniformScaling)
    return data(A) + S, Δ->(ones(size(A)), nothing)
end
