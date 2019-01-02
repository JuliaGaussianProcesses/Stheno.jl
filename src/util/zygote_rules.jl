using Zygote, IRTools
using Zygote: @adjoint, _forward

import Base: map
import Distances: pairwise, colwise
import LinearAlgebra: \, /
import FillArrays: Fill, AbstractFill, getindex_value

# Hack at Zygote to make Fills work properly.
@adjoint Fill(x, sz::Tuple{Vararg}) = Fill(x, sz), Δ->(sum(Δ), nothing)
@adjoint Fill(x, sz::Int) = Fill(x, sz), Δ->(sum(Δ), nothing)


const AbstractFillVec{T} = AbstractFill{T, 1}
const AbstractFillMat{T} = AbstractFill{T, 2}

@adjoint function broadcasted(op, r::AbstractFill{<:Real})
    y, _back = Zygote.forward(op, getindex_value(r))
    back(Δ::AbstractFill) = (nothing, Fill(_back(getindex_value(Δ))[1], size(r)))
    back(Δ::AbstractArray) = (nothing, getindex.(_back.(Δ), 1))
    return Fill(y, size(r)), back
end

for array_type in [:AbstractFillVec, :AbstractFillMat]
    @eval @adjoint function broadcasted(::typeof(+), a::$array_type, b::$array_type)
        return broadcasted(+, a, b), Δ->(nothing, Δ, Δ)
    end
    @eval @adjoint function broadcasted(::typeof(*), a::$array_type, b::$array_type)
        return broadcasted(*, a, b), Δ->(nothing, Δ .* b, Δ .* a)
    end
end

@adjoint function sqeuclidean(x::AbstractVector, y::AbstractVector)
    δ = x .- y
    return sum(abs2, δ), function(Δ::Real)
        x̄ = (2 * Δ) .* δ
        return x̄, -x̄
    end
end

@adjoint function colwise(s::SqEuclidean, x::AbstractMatrix, y::AbstractMatrix)
    return colwise(s, x, y), function (Δ::AbstractVector)
        x̄ = 2 .* Δ' .* (x .- y)
        return nothing, x̄, -x̄
    end
end

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
    return pairwise(s, x, y), function(Δ)
        x̄ = 2 .* (reshape(sum(Δ; dims=2), :) .* x .- Δ * y)
        ȳ = 2 .* (reshape(sum(Δ; dims=1), :)  .* y .- Δ' * x)
        return nothing, x̄, ȳ
    end
end

@adjoint function pairwise(s::SqEuclidean, x::AbstractVector)
    return pairwise(s, x), function(Δ)
        return nothing, 4 .* (reshape(sum(Δ; dims=1), :) .* x .- Δ * x)
    end
end

@adjoint function \(A::AbstractMatrix, B::AbstractVecOrMat)
    Y = A \ B
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

@adjoint function /(A::AbstractMatrix, B::AbstractMatrix)
    Y = A / B
    return Y, function(Ȳ)
        Ā = Ȳ / B'
        return (Ā, -Y' * Ā)
    end
end

@adjoint Symmetric(A::AbstractMatrix) = Symmetric(A), Δ->(symmetric_back(Δ),)

@adjoint function chol(Σ::Symmetric)
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

@adjoint diag(A::AbstractMatrix) = diag(A), Δ->(Diagonal(Δ),)

@adjoint function logdet(U::StridedTriangular)
    return logdet(U), Δ->(UpperTriangular(Matrix(Diagonal(Δ ./ diag(U)))),)
end

@adjoint +(A::AbstractMatrix, S::UniformScaling) = (A + S, Δ->(ones(size(A)), nothing))

# @adjoint function map(f, x...)
#     @show f, x
#     y_pairs = map((x...)->Zygote.forward(f, x...), x...)
#     y = [y_pair[1] for y_pair in y_pairs]
#     return y, function(Δ)
#         out_back = map((δ, (y, back))->back(δ), Δ, y_pairs)
#         xs = (nothing, map(n->[p[n] for p in out_back], 1:length(x))...)
#     end
# end

# Get rid of fusion when we're broadcasting because it's a bit of a pain and not
# _completely_ essential at this stage.
@adjoint function broadcasted(f, x...)
    y_pairs = materialize(broadcasted((x...)->Zygote.forward(f, x...), x...))
    y = [y_pair[1] for y_pair in y_pairs] 
    return y, function(Δ)
        out_back = broadcast((δ, (y, back))->back(δ), Δ, y_pairs)
        xs = (nothing, map(n->unbroadcast(x[n], [p[n] for p in out_back]), 1:length(x))...)
    end
end

# Shamelessly stolen from Zygote.
trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

# Shamelessly stolen from Zygote.
function unbroadcast(x::AbstractArray, Δ)
    return size(x) == size(Δ) ? Δ :
        length(x) == length(Δ) ? trim(x, Δ) :
            trim(x, sum(
                Δ,
                dims=ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ))),
            )
        )
end

unbroadcast(x::AbstractArray, Δ::AbstractArray{Nothing}) = trim(x, Δ)

# Shamelessly stolen from Zygote.
unbroadcast(x::Union{Number, Ref}, Δ) = sum(Δ)

unbroadcast(x, Δ::AbstractArray{Nothing}) = nothing
