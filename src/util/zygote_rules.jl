using Zygote, IRTools
using Zygote: @adjoint

import Distances: pairwise
import LinearAlgebra: \, /

@adjoint function pairwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
    return pairwise(s, x, y), function(Δ)
        x̄ = 2 .* (x * Diagonal(sum(Δ; dims=2)) .- y * Δ')
        ȳ = 2 .* (y * Diagonal(sum(Δ; dims=1)) .- x * Δ)
        return nothing, x̄, ȳ
    end
end

@adjoint function pairwise(s::SqEuclidean, X::AbstractMatrix)
    D = pairwise(s, X)
    return D, Δ->(nothing, 4 * (X * (Diagonal(reshape(sum(Δ; dims=1), :)) - Δ)))
end

@adjoint function pairwise(s::SqEuclidean, x::AbstractVector, y::AbstractVector)
    D = pairwise(s, x, y)
    return D, function(Δ)
        x̄ = 2 .* (reshape(sum(Δ; dims=2), :) .* x .- Δ * y)
        ȳ = 2 .* (reshape(sum(Δ; dims=1), :)  .* y .- Δ' * x)
        return nothing, x̄, ȳ
    end
end

@adjoint function pairwise(s::SqEuclidean, x::AbstractVector)
    D = pairwise(s, x)
    return D, function(Δ)
        return nothing, 4 .* (reshape(sum(Δ; dims=1), :) .* x .- Δ * x)
    end
end

@adjoint function \(A::AbstractMatrix, B::AbstractVecOrMat)
    println("Attempting to backsolve A \\ B")
    Y = A \ B
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

@adjoint function /(A::AbstractMatrix, B::AbstractMatrix)
    println("Attempting to backsolve A / B")
    Y = A / B
    return Y, function(Ȳ)
        Ā = Ȳ / B'
        return (Ā, -Y' * Ā)
    end
end

@adjoint function Symmetric(A::AbstractMatrix)
    println("Attemping to Symmetrise")
    @show size(A)
    return Symmetric(A), Δ->(symmetric_back(Δ),)
end

@adjoint function chol(Σ::AbstractMatrix)
    println("Attempting to chol")
    U = chol(Σ)
    return U, function(Ū)
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] *= 0.5
        end
        return (UpperTriangular(Σ̄),)
    end
end

@adjoint function diag(A::AbstractMatrix)
    println("Attempting to diag")
    return diag(A), Δ->(Diagonal(Δ),)
end

@adjoint function logdet(U::StridedTriangular)
    return logdet(U), Δ->(UpperTriangular(Matrix(Diagonal(Δ ./ diag(U)))),)
end

@adjoint +(A::AbstractMatrix, S::UniformScaling) = (A + S, Δ->(ones(size(A)), nothing))
