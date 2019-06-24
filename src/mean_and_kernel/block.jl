"""
    BlockMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct BlockMean{Tμ<:AbstractVector{<:MeanFunction}} <: MeanFunction
    μ::Tμ
end
function ew(m::BlockMean, x::BlockData)
    blks = map((μ, blk)->ew(μ, blk), m.μ, blocks(x))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
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

function ew(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    blks = map((k, b, b′)->ew(k, b, b′), diag(k.ks), blocks(x), blocks(x′))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end
function pw(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    return Matrix(_pw(k.ks, blocks(x), permutedims(blocks(x′))))
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
    blks = map((k, b, b′)->ew(k, b, b′), diag(k.ks), blocks(x), blocks(x′))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end
function pw(k::BlockKernel, x::BlockData, x′::BlockData)
    return Matrix(_pw(k.ks, blocks(x), permutedims(blocks(x′))))
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

        # Reduce over appropriate dimensions manually because sum doesn't work... :S
        δ_x = Vector{Any}(undef, length(x_blks))
        δ_x′ = Vector{Any}(undef, length(x′_blks))

        for p in 1:length(x_blks)
            δ_x[p] = Δ_x[p, 1]
            for q in 2:length(x′_blks)
                δ_x[p] = Zygote.accum(δ_x[p], Δ_x[p, q])
            end
        end

        for q in 1:length(x′_blks)
            δ_x′[q] = Δ_x′[1, q]
            for p in 2:length(x_blks)
                δ_x′[q] = Zygote.accum(δ_x′[q], Δ_x′[p, q])
            end
        end

        return Δ_ks, δ_x, permutedims(δ_x′)
    end
    return Y, back
end
