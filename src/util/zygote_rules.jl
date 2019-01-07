using Zygote, IRTools, Random, Base.Threads
using Zygote: @adjoint, _forward, literal_getproperty

import Base: map, getfield, getproperty, sum
import Distances: pairwise, colwise
import LinearAlgebra: \, /, cholesky, copytri!
import FillArrays: Fill, AbstractFill, getindex_value
import Base.Broadcast: broadcasted, materialize

@nograd MersenneTwister, propertynames

@adjoint Fill(x, sz::Integer...) = Fill(x, sz...), Δ->(sum(Δ), map(_->nothing, sz)...)
# @adjoint Fill(x, sz::Int) = Fill(x, sz), Δ->(sum(Δ), nothing)
# @adjoint Fill(x, p::Int, q::Int) = Fill(x, p, q), Δ->(sum(Δ), nothing, nothing)

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

symmetric_back(Δ) = UpperTriangular(Δ) + LowerTriangular(Δ)' - Diagonal(Δ)
symmetric_back(Δ::UpperTriangular) = Δ
@adjoint function Symmetric(A::AbstractMatrix)
    back(Δ::AbstractMatrix) = (symmetric_back(Δ),)
    back(Δ::NamedTuple) = (symmetric_back(Δ.data),)
    return Symmetric(A), back
end

@adjoint function cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}})
    C = cholesky(Σ)
    U = C.U
    return C, function(Δ)
        Ū = Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄),)
    end
end

@adjoint SymmetricToeplitz(v) = SymmetricToeplitz(v), Δ::NamedTuple->(Δ.vc,)

@adjoint function cholesky(T::SymmetricToeplitz)

    # Allocate memory for output L and temporaries V, a, and b.
    v, N = T.vc, size(T, 1)
    L = Matrix{eltype(T)}(undef, N, N)
    a = Vector{eltype(T)}(undef, N)

    # Initialise L and V.
    L[:, 1] .= v ./ sqrt(v[1]) # 1
    v_ = L[:, 1] # 2

    # Iterate over the columns of L.
    @inbounds for n in 1:N-1
        a[n] = v_[n+1] / L[n, n] # 3
        b = sqrt(1 - a[n]^2) # 4

        for n′ in (n+1):N
            v_[n′] = (v_[n′] - a[n] * L[n′-1, n]) / b # 5
            L[n′, n+1] = -a[n] * v_[n′] + b * L[n′-1, n] # 6
        end
    end

    back(C̄::NamedTuple{(:uplo, :info, :factors)}) = back(C̄.factors)
    function back(L̄_in::AbstractMatrix)
        v̄_ = zero(v_)

        # Allocate memory for L̄ to avoid modifying the sensitivity provided.
        L̄ = Matrix{eltype(L)}(undef, N, N)
        copyto!(L̄, L̄_in)

        @inbounds for n in reverse(1:N-1)
            b, b̄, ā = sqrt(1 - a[n]^2), zero(eltype(a)), zero(eltype(a))

            for n′ in reverse((n+1):N)
                ā -= L̄[n′, n+1] * v_[n′] # 6
                v̄_[n′] -= L̄[n′, n+1] * a[n] # 6
                b̄ += L̄[n′, n+1] * L[n′-1, n] # 6
                L̄[n′-1, n] += L̄[n′, n+1] * b # 6

                v_[n′] = b * v_[n′] + a[n] * L[n′-1, n] # invert 5

                ā -= v̄_[n′] * L[n′-1, n] / b # 5
                L̄[n′-1, n] -= v̄_[n′] * a[n] / b # 5
                b̄ += v̄_[n′] * (a[n] * L[n′-1, n] - v_[n′]) / b^2 # 5
                v̄_[n′] = v̄_[n′] / b # 5
            end

            ā -= b̄ * a[n] / b # 4

            v̄_[n+1] += ā / L[n, n] # 3
            L̄[n, n] -= ā * v_[n+1] / L[n, n]^2 # 3
        end

        L̄[:, 1] .+= v̄_ # 2

        v̄ = L̄[:, 1] ./ sqrt(v[1]) # 1
        v̄[1] -= sum(n->L̄[n, 1] * v[n], 1:N) / (2 * sqrt(v[1])^3) # 1
        return ((vc=v̄, tmp=nothing, dft=nothing, vcvr_dft=nothing),)
    end
    return Cholesky(L, 'L', 0), back
end

# Various sensitivities for `literal_getproperty`, depending on the 2nd argument.
@adjoint function literal_getproperty(C::Cholesky, ::Val{:uplo})
    return literal_getproperty(C, Val(:uplo)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:info})
    return literal_getproperty(C, Val(:info)), function(Δ)
        return ((uplo=nothing, info=nothing, factors=nothing),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:factors})
    return literal_getproperty(C, Val(:factors)), function(Δ)
        error("@adjoint not implemented for :factors. (I couldn't make it work...)")
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:U})
    return literal_getproperty(C, Val(:U)), function(Δ)
        Δ_factors = C.uplo == 'U' ? UpperTriangular(Δ) : LowerTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end
@adjoint function literal_getproperty(C::Cholesky, ::Val{:L})
    return literal_getproperty(C, Val(:L)), function(Δ)
        Δ_factors = C.uplo == 'L' ? LowerTriangular(Δ) : UpperTriangular(copy(Δ'))
        return ((uplo=nothing, info=nothing, factors=Δ_factors),)
    end
end

# Return something that the cholesky knows how to work with.
@adjoint function logdet(C::Cholesky)
    return logdet(C), function(Δ)
        return ((info=nothing, uplo=nothing, factors=Diagonal(2 .* Δ ./ diag(C.factors))),)
    end
end


@adjoint diag(A::AbstractMatrix) = diag(A), Δ->(Diagonal(Δ),)

@adjoint function +(A::AbstractMatrix, S::UniformScaling)
    return A + S, Δ->(Δ, (λ=sum(view(Δ, diagind(Δ))),))
end

# @adjoint function map(f, x...)
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
