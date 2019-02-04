using Zygote, IRTools, Random, ToeplitzMatrices
using Zygote: @adjoint, _forward, literal_getproperty
import Zygote: accum

import Base: map, getfield, getproperty, sum
import Distances: pairwise, colwise
import LinearAlgebra: \, /, cholesky, copytri!
import FillArrays: Fill, AbstractFill, getindex_value, Zeros, Ones
import Base.Broadcast: broadcasted, materialize

import StatsFuns: log1pexp, logistic, logexpm1

@nograd MersenneTwister, propertynames

# Adjoints for FillArrays.jl. concrete types.
@adjoint function Fill(x, sz::Integer...)
    back(Δ::Nothing) = (nothing, map(_->nothing, sz)...)
    back(Δ::AbstractArray) = (sum(Δ), map(_->nothing, sz)...)
    function back(Δ::NamedTuple{(:value, :axes)})
        return (Δ.value isa Nothing ? nothing : sum(Δ.value), map(_->nothing, sz)...)
    end
    return Fill(x, sz...), back
end
@adjoint Zeros{T}(sz::Integer...) where {T} = Zeros{T}(sz...), Δ->(map(_->nothing, sz)...,)
@adjoint Ones{T}(sz::Integer...) where {T} = Ones{T}(sz...), Δ->(map(_->nothing, sz)...,)

@adjoint function broadcasted(op, r::AbstractFill{<:Real})
    y, _back = Zygote.forward(op, getindex_value(r))
    back(Δ::AbstractFill) = (nothing, Fill(_back(getindex_value(Δ))[1], size(r)))
    back(Δ::AbstractArray) = (nothing, getindex.(_back.(Δ), 1))
    return Fill(y, size(r)), back
end

for array_type in [:(AbstractFill{T, 1} where T), :(AbstractFill{T, 2} where T)]
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

@adjoint function Diagonal(x::AbstractVector)
    back(Δ::NamedTuple) = (Δ.diag,)
    back(Δ::AbstractMatrix) = (diag(Δ),)
    return Diagonal(x), back
end

@adjoint function log1pexp(x::Real)
    return log1pexp(x), Δ->(Δ * (x < 18.0 ? logistic(x) : x < 33.3 ? 1 - exp(-x) : 1),)
end

@adjoint function log1pexp(x::Float32)
    return log1pexp(x), Δ->(Δ * (x < 9f0 ? logistic(x) : x < 16f0 ? 1 - exp(-x) : 1),)
end
