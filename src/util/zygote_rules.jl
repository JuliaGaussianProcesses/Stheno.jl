using Zygote
using Zygote: @adjoint, literal_getproperty
import Zygote: accum

import Distances: pairwise, colwise

@nograd MersenneTwister, propertynames, broadcast_shape

function accum(D::Diagonal{T}, B::AbstractMatrix) where {T}
    A = Matrix{Union{T, Nothing}}(undef, size(D))
    A[diagind(A)] .= D.diag
    return accum(A, B)
end
accum(A::AbstractMatrix, D::Diagonal) = accum(D, A)
accum(A::Diagonal, B::Diagonal) = Diagonal(accum(diag(A), diag(B)))

@adjoint function colwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
    return colwise(s, x, y), function (Δ::AbstractVector)
        x̄ = 2 .* Δ' .* (x .- y)
        return nothing, x̄, -x̄
    end
end

@adjoint function pairwise(s::SqEuclidean, X::AbstractMatrix; dims=2)
    D = pairwise(s, X; dims=dims)
    return D, function(Δ)
        d1, d2 = Diagonal(vec(sum(Δ; dims=1))), Diagonal(vec(sum(Δ; dims=2)))
        return (nothing, X * (2 .* (d1 .+ d2 .- Δ .- Δ')))
    end
end

@adjoint function colwise(s::Euclidean, x::AbstractMatrix, y::AbstractMatrix)
    d = colwise(s, x, y)
    return d, function (Δ::AbstractVector)
        x̄ = (Δ ./ d)' .* (x .- y)
        return nothing, x̄, -x̄
    end
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
    @assert dims == 2
    D, back = Zygote.forward((X, Y)->pairwise(SqEuclidean(), X, Y; dims=2), X, Y)
    D .= sqrt.(D)
    return D, Δ -> (nothing, back(Δ ./ (2 .* D))...)
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix; dims=2)
    @assert dims == 2
    D, back = Zygote.forward(X->pairwise(SqEuclidean(), X; dims=2), X)
    D .= sqrt.(D)
    return D, function(Δ)
        Δ = Δ ./ (2 .* D)
        Δ[diagind(Δ)] .= 0
        return (nothing, first(back(Δ)))
    end
end

@adjoint function literal_getproperty(C::Cholesky, ::Val{:factors})
    error("@adjoint not implemented for :factors as is unsafe.")
    return literal_getproperty(C, Val(:factors)), function(Δ)
        error("@adjoint not implemented for :factors. (I couldn't make it work...)")
    end
end

import LinearAlgebra: HermOrSym, diag, Diagonal

diag(S::Symmetric{T, <:Diagonal{T}} where T) = S.data.diag
Zygote._symmetric_back(Δ::Diagonal) = Δ

# Diagonal matrices are always symmetric...
cholesky(A::HermOrSym{T, <:Diagonal{T}} where T) = cholesky(Diagonal(diag(A)))
