export BlockMean, BlockKernel, BlockCrossKernel


"""
    BlockMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct BlockMean <: MeanFunction
    μ::Vector
end
BlockMean(μs::Vararg{<:MeanFunction}) = BlockMean([μs...])
# _map(μ::BlockMean, X::BlockData) = BlockVector(map.(μ.μ, blocks(X)))
_map(m::BlockMean, X::BlockData) = BlockVector([map(m, x) for (m, x) in zip(m.μ, blocks(X))])


"""
    BlockCrossKernel <: CrossKernel

A cross kernel comprising lots of other kernels.
"""
struct BlockCrossKernel <: CrossKernel
    ks::Matrix
end
BlockCrossKernel(ks::AbstractVector) = BlockCrossKernel(reshape(ks, length(ks), 1))
function BlockCrossKernel(ks::Adjoint{T, AbstractVector{T}} where T)
    return BlockCrossKernel(reshape(ks, 1, length(ks)))
end

# Binary methods.
function _map(k::BlockCrossKernel, X::BlockData, X′::BlockData)
    return BlockVector(map.(diag(k.ks), blocks(X), blocks(X′)))
end
function _pw(k::BlockCrossKernel, X::BlockData, X′::BlockData)
    return BlockMatrix(broadcast(pairwise, k.ks, blocks(X), reshape(blocks(X′), 1, :)))
end

# IS IT EVEN REASONABLE TO DEFINE THIS?
# _map(k::BlockCrossKernel, X::BlockData) = BlockVector(map.(diag(k.ks), blocks(X)))


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
struct BlockKernel <: Kernel
    ks_diag::Vector{<:Kernel}
    ks_off::Matrix{<:CrossKernel}
end

# Binary methods.
function _map(k::BlockKernel, X::BlockData, X′::BlockData)
    return BlockVector(map.(k.ks_diag, blocks(X), blocks(X′)))
end
function _pw(k::BlockKernel, X::BlockData, X′::BlockData)
    bX, bX′ = blocks(X), blocks(X′)
    Ω = BlockArray(undef_blocks, AbstractMatrix{Float64}, length.(bX), length.(bX′))
    for q in eachindex(k.ks_diag), p in eachindex(k.ks_diag)
        if p == q
            setblock!(Ω, pairwise(k.ks_diag[p], bX[p], bX′[p]), p, p)
        elseif p < q
            setblock!(Ω, pairwise(k.ks_off[p, q], bX[p], bX′[q]), p, q)
        else
            setblock!(Ω, pairwise(k.ks_off[q, p], bX[p], bX′[q]), p, q)
        end
    end
    return Ω
end

# Unary methods.
Base.adjoint(z::Zeros{T}) where {T} = Zeros{T}(reverse(size(z)))
_map(k::BlockKernel, X::BlockData) = BlockVector(map.(k.ks_diag, blocks(X)))
function _pw(k::BlockKernel, X::BlockData)
    bX = blocks(X)
    Σ = BlockArray(undef_blocks, AbstractMatrix{Float64}, length.(bX), length.(bX))
    for q in eachindex(k.ks_diag)
        setblock!(Σ, unbox(pairwise(k.ks_diag[q], bX[q])), q, q)
        for p in 1:q-1
            setblock!(Σ, pairwise(k.ks_off[p, q], bX[p], bX[q]), p, q)
            setblock!(Σ, getblock(Σ, p, q)', q, p)
        end
    end
    return LazyPDMat(Symmetric(Σ))
end
