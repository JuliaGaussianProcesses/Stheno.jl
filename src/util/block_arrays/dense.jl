# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

import Base: +, *, size, getindex, eltype, copy, \, vec, getproperty, zero
import LinearAlgebra: UpperTriangular, LowerTriangular, logdet, Symmetric, transpose,
    adjoint, AdjOrTrans, AdjOrTransAbsMat, cholesky!, logdet, ldiv!, mul!, logabsdet

"""
    blocksizes(X::AbstractBlockVecOrMat)
    blocksizes(X::AbstractArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. If `d` not
specified, returns a `Tuple` over blocksizes over each dimension.
"""
blocksizes(X::AbstractBlockArray, d::Int) = blocklengths(axes(X, d))
@nograd blocksizes

cumulsizes(X::AbstractBlockArray, d::Int) = vcat(1, cumsum(blocksizes(X, d)) .+ 1)
@nograd cumulsizes

nblocks(X::BlockArray, d::Int) = size(X.blocks, d)

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = blocksizes(A, 2) == blocksizes(B, 1)

function rrule(::typeof(BlockArrays.mortar), _blocks::AbstractArray)
    function mortar_pullback(Δ::Composite)
        return (NO_FIELDS, Δ.blocks, )
    end
    function mortar_pullback(Δ::BlockArray)
        return mortar_pullback((blocks = Δ.blocks, axes=nothing))
    end
    return BlockArrays.mortar(_blocks), mortar_pullback
end

ZygoteRules.@adjoint function BlockArrays.Array(X::BlockArray)
    function Array_pullback(Δ::Array)
        ΔX = (blocks=BlockArray(Δ, axes(X)).blocks, axes=nothing)
        return (ΔX,)
    end
    return Array(X), Array_pullback
end
