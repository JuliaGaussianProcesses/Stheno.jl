# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

"""
    blocksizes(X::AbstractBlockVecOrMat)
    blocksizes(X::AbstractArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. If `d` not
specified, returns a `Tuple` over blocksizes over each dimension.
"""
blocksizes(X::AbstractBlockArray, d::Int) = blocklengths(axes(X, d))
ChainRulesCore.@non_differentiable blocksizes(::Any, ::Any)

cumulsizes(X::AbstractBlockArray, d::Int) = vcat(1, cumsum(blocksizes(X, d)) .+ 1)
ChainRulesCore.@non_differentiable cumulsizes(::Any, ::Any)

nblocks(X::BlockArray, d::Int) = size(X.blocks, d)

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::BlockVecOrMat, B::BlockVecOrMat) = blocksizes(A, 2) == blocksizes(B, 1)

function ChainRulesCore.rrule(::typeof(BlockArrays.mortar), _blocks::AbstractArray)
    y = BlockArrays.mortar(_blocks)
    Ty = typeof(y)
    function mortar_pullback(Δ::Tangent)
        return (NoTangent(), Δ.blocks)
    end
    function mortar_pullback(Δ::BlockArray)
        return mortar_pullback(Tangent{Ty}(; blocks = Δ.blocks, axes=NoTangent()))
    end
    return y, mortar_pullback  
end

# A hook to which I can attach an rrule without commiting type-piracy against BlockArrays.
_collect(X::BlockArray) = Array(X)

function ChainRulesCore.rrule(::typeof(_collect), X::BlockArray)
    function Array_pullback(Δ::Array)
        ΔX = Tangent{Any}(blocks=BlockArray(Δ, axes(X)).blocks, axes=NoTangent())
        return (NoTangent(), ΔX)
    end
    return Array(X), Array_pullback
end
