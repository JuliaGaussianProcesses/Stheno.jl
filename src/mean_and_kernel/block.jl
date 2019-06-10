Base.zero(A::AbstractArray{<:AbstractArray}) = zero.(A)


"""
    BlockMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct BlockMean{Tμ<:AbstractVector{<:MeanFunction}} <: MeanFunction
    μ::Tμ
end
BlockMean(μs::Vararg{<:MeanFunction}) = BlockMean([μs...])
_map(m::BlockMean, x::BlockData) = BlockVector(map((μ, blk)->map(μ, blk), m.μ, blocks(x)))


"""
    BlockCrossKernel <: CrossKernel

A cross kernel comprising lots of other kernels.
"""
struct BlockCrossKernel{Tks<:Matrix{<:CrossKernel}} <: CrossKernel
    ks::Tks
end
BlockCrossKernel(ks::AbstractVector) = BlockCrossKernel(reshape(ks, length(ks), 1))
function BlockCrossKernel(ks::Adjoint{T, <:AbstractVector{T}} where T)
    return BlockCrossKernel(reshape(ks, 1, length(ks)))
end

# Binary methods.
function _map(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    return BlockVector(map((k, b, b′)->map(k, b, b′), diag(k.ks), blocks(x), blocks(x′)))
end
function _pw(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    blks = pw.(k.ks, blocks(x), permutedims(blocks(x′)))
    return _BlockArray(blks, _get_block_sizes(blks)...)
end
_pw(k::BlockCrossKernel, x::BlockData, x′::AV) = _pw(k, x, BlockData([x′]))
_pw(k::BlockCrossKernel, x::AV, x′::BlockData) = _pw(k, BlockData([x]), x′)


"""
    BlockKernel <: Kernel

A kernel comprising lots of other kernels. This is represented as a matrix whose diagonal
elements are `Kernels`, and whose off-diagonal elements are `CrossKernel`s. In the absence
of determining at either either compile- or construction-time whether or not this actually
constitutes a valid Mercer kernel, we take the construction of this type to be a promise on
the part of the caller that the thing they are constructing does indeed constitute a valid
Mercer kernel.

`ks_diag` represents the kernels on the diagonal of this matrix-valued kernel, and `ks_off`
represents the elements in the rest of the matrix. Only the upper-triangle will actually
be used.
"""
struct BlockKernel{Tks<:Matrix{<:CrossKernel}} <: Kernel
    ks::Tks
end

# Binary methods.
function _map(k::BlockKernel, x::BlockData, x′::BlockData)
    items = zip(diag(k.ks), blocks(x), blocks(x′))
    return BlockVector([map(k, blk, blk′) for (k, blk, blk′) in items])
end
function _pw(k::BlockKernel, x::BlockData, x′::BlockData)
    x_items, x′_items = enumerate(blocks(x)), enumerate(blocks(x′))
    blks = [pw(k.ks[p, q], x, x′) for (p, x) in x_items, (q, x′) in x′_items]
    return _BlockArray(blks, _get_block_sizes(blks)...)
end

# Unary methods.
_map(k::BlockKernel, x::BlockData) = _map(k, x, x)
_pw(k::BlockKernel, x::BlockData) = _pw(k, x, x)
