using Zygote
import Zygote: accum

function accum(D::Diagonal{T}, B::AbstractMatrix) where {T}
    A = Matrix{Union{T, Nothing}}(undef, size(D))
    A[diagind(A)] .= D.diag
    return accum(A, B)
end
accum(A::AbstractMatrix, D::Diagonal) = accum(D, A)
accum(A::Diagonal, B::Diagonal) = Diagonal(accum(diag(A), diag(B)))

#
# Some very specific broadcasting hacks while Zygote has crappy broadcasting.
#

import Base.Broadcast: broadcasted

function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(-), x::AbstractArray)
    function broadcasted_minus_pullback(Δ)
        return (NoTangent(), NoTangent(), .-Δ)
    end
    return .-x, broadcasted_minus_pullback
end

function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(exp), x::AbstractArray)
    y = exp.(x)
    return y, Δ->(NoTangent(), NoTangent(), Δ .* y)
end
