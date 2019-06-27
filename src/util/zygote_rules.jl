using Zygote, IRTools, Random, ToeplitzMatrices
using Zygote: @adjoint, literal_getproperty
import Zygote: accum

import Distances: pairwise, colwise

@nograd MersenneTwister, propertynames

function accum(D::Diagonal{T}, B::AbstractMatrix) where {T}
    A = Matrix{Union{T, Nothing}}(undef, size(D))
    A[diagind(A)] .= D.diag
    return accum(A, B)
end
accum(A::AbstractMatrix, D::Diagonal) = accum(D, A)
accum(A::Diagonal, B::Diagonal) = Diagonal(accum(diag(A), diag(B)))

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
        x̄ = 2 .* (x * Diagonal(vec(sum(Δ; dims=2))) .- y * Δ')
        ȳ = 2 .* (y * Diagonal(vec(sum(Δ; dims=1))) .- x * Δ)
        return nothing, x̄, ȳ
    end
end

@adjoint function pairwise(s::SqEuclidean, X::AbstractMatrix)
    D = pairwise(s, X)
    return D, function(Δ)
        d1, d2 = Diagonal(vec(sum(Δ; dims=1))), Diagonal(vec(sum(Δ; dims=2)))
        return (nothing, X * (2 .* (d1 .+ d2 .- Δ .- Δ')))
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

    back(C̄::NamedTuple) = back(C̄.factors)
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

@adjoint function literal_getproperty(C::Cholesky, ::Val{:factors})
    error("@adjoint not implemented for :factors as is unsafe.")
    return literal_getproperty(C, Val(:factors)), function(Δ)
        error("@adjoint not implemented for :factors. (I couldn't make it work...)")
    end
end

using Base: Generator

@adjoint Generator(f, iter) = Generator(f, iter), Δ::NamedTuple{(:f, :iter)}->(Δ.f, Δ.iter)
