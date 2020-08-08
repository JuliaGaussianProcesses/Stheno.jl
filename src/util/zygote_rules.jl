using Zygote
using Zygote: @adjoint, literal_getproperty
import Zygote: accum

import Distances: pairwise, colwise
const dtol = 1e-12 # threshold value for precise recalculation of distances

@nograd MersenneTwister, propertynames, broadcast_shape

function accum(D::Diagonal{T}, B::AbstractMatrix) where {T}
    A = Matrix{Union{T, Nothing}}(undef, size(D))
    A[diagind(A)] .= D.diag
    return accum(A, B)
end
accum(A::AbstractMatrix, D::Diagonal) = accum(D, A)
accum(A::Diagonal, B::Diagonal) = Diagonal(accum(diag(A), diag(B)))

@adjoint function ZygoteRules.literal_getproperty(C::Cholesky, ::Val{:factors})
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



#
# Some very specific broadcasting hacks while Zygote has crappy broadcasting.
#

import Base.Broadcast: broadcasted

function rrule(::typeof(broadcasted), ::typeof(-), x::AbstractArray)
    function broadcasted_minus_pullback(Δ)
        return (NO_FIELDS, DoesNotExist(), .-Δ)
    end
    return .-x, broadcasted_minus_pullback
end

function rrule(::typeof(broadcasted), ::typeof(exp), x::AbstractArray)
    y = exp.(x)
    return y, Δ->(NO_FIELDS, DoesNotExist(), Δ .* y)
end
