# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

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
