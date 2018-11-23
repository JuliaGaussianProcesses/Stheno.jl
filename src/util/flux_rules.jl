using Flux, Flux.Tracker
using Flux.Tracker: track, @grad, TrackedVecOrMat, data

import Distances: pairwise
import LinearAlgebra: cholesky, UpperTriangular, logdet, \, /

# Binary pairwise.
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

# Unary pairwise.
pairwise(s::SqEuclidean, X::TrackedMatrix) = track(pairwise, s, X)
@grad function pairwise(s::SqEuclidean, X::AbstractMatrix)
    D = pairwise(s, data(X))
    return D, function(Δ)
        return nothing, 4 * (X * (Diagonal(reshape(sum(Δ; dims=1), :)) - Δ))
    end
end

import Base: *
*(A::TrackedMatrix, B::Diagonal) = track(*, A, B)
*(A::Diagonal, B::TrackedMatrix) = track(*, A, B)

# Y = A \ B
\(A::TrackedMatrix, B::TrackedVecOrMat) = track(\, A, B)
\(A::AbstractMatrix, B::TrackedVecOrMat) = track(\, A, B)
\(A::TrackedMatrix, B::AbstractVecOrMat) = track(\, A, B)
@grad function \(A::AbstractMatrix, B::AbstractVector)
    Y = data(A) \ data(B)
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

# # Y = A / B
# /(A::TrackedVecOrMat, B::TrackedMatrix) = track(/, A, B)
# /(A::AbstractVecOrMat, B::TrackedMatrix) = track(/, A, B)
# /(A::TrackedVecOrMat, B::AbstractMatrix) = track(/, A, B)
# @grad function /(A::AbstractVecOrMat, B::AbstractMatrix)
#     Y = data(A) / data(B)
#     return Y, function(Ȳ)
#         Ā = Ȳ / B'
#         return (Ā, -Y' * Ā)
#     end
# end



