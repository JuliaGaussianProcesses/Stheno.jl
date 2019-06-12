Base.zero(A::AbstractArray{<:AbstractArray}) = zero.(A)


"""
    BlockMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct BlockMean{Tμ<:AbstractVector{<:MeanFunction}} <: MeanFunction
    μ::Tμ
end
BlockMean(μs::Vararg{<:MeanFunction}) = BlockMean([μs...])
function ew(m::BlockMean, x::BlockData)
    return BlockVector(map((μ, blk)->ew(μ, blk), m.μ, blocks(x)))
end

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
function ew(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    return BlockVector(map((k, b, b′)->ew(k, b, b′), diag(k.ks), blocks(x), blocks(x′)))
end
function pw(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    return _pw(k.ks, blocks(x), permutedims(blocks(x′)))
end

pw(k::BlockCrossKernel, x::BlockData, x′::AV) = pw(k, x, BlockData([x′]))
pw(k::BlockCrossKernel, x::AV, x′::BlockData) = pw(k, BlockData([x]), x′)


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
function ew(k::BlockKernel, x::BlockData, x′::BlockData)
    return BlockVector(map((k, b, b′)->ew(k, b, b′), diag(k.ks), blocks(x), blocks(x′)))
end
function pw(k::BlockKernel, x::BlockData, x′::BlockData)
    return _pw(k.ks, blocks(x), permutedims(blocks(x′)))
end

# Unary methods.
ew(k::BlockKernel, x::BlockData) = ew(k, x, x)
pw(k::BlockKernel, x::BlockData) = pw(k, x, x)


#
# Helper for pw.
#

function _pw(ks, x_blks, x′_blks)
    blks = pw.(ks, x_blks, x′_blks)
    return _BlockArray(blks, _get_block_sizes(blks)...)
end
@adjoint function _pw(ks, x_blks, x′_blks)
    blk_backs = broadcast((k, x, x′)->Zygote.forward(pw, k, x, x′), ks, x_blks, x′_blks)
    blks, backs = first.(blk_backs), last.(blk_backs)
    Y = _BlockArray(blks, _get_block_sizes(blks)...)

    function back(Δ::BlockMatrix)

        Δ_k_x_x′ = broadcast((back, blk)->back(blk), backs, Δ.blocks)
        Δ_ks, Δ_x, Δ_x′ = first.(Δ_k_x_x′), getindex.(Δ_k_x_x′, 2), getindex.(Δ_k_x_x′, 3)

        # zero out nothings.
        for p in 1:length(x_blks), q in 1:length(x′_blks)
            if Δ_x[p, q] === nothing
                Δ_x[p, q] = zeros(size(x_blks[p]))
            end
            if Δ_x′[p, q] === nothing
                Δ_x′[p, q] = zeros(size(x′_blks[q]))
            end
        end

        # Reduce over appropriate dimensions manually because sum doesn't work... :S
        δ_x, δ_x′ = zero.(Δ_x[:, 1]), zero.(Δ_x′[1, :])
        for p in 1:length(x_blks), q in 1:length(x′_blks)
            δ_x[p] += Δ_x[p, q]
            δ_x′[q] += Δ_x′[p, q]
        end

        return Δ_ks, δ_x, δ_x′
    end
    back(Δ::AbstractMatrix) = back(BlockArray(Δ, _get_block_sizes(blks)...))

    return Y, back
end
