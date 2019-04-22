# These implementations are slow, and are only used because Zygote doesn't currently support
# another way to achieve the required behaviour.
function unary_make_vec(foo, fs, xs)
    y = foo(fs[1], xs[1])
    for n in 2:length(xs)
        y = vcat(y, foo(fs[n], xs[n]))
    end
    return y
end
function binary_make_vec(foo, fs, xs, x′s)
    y = foo(fs[1], xs[1], x′s[1])
    for n in 2:length(xs)
        y = vcat(y, foo(fs[n], xs[n], x′s[n]))
    end
    return y
end
function binary_make_mat(foo, fs, xs, x′s)
    Y = unary_make_vec((f, x)->foo(f, x, x′s[1]), fs[:, 1], xs)
    for n in 2:length(x′s)
        Y = hcat(Y, unary_make_vec((f, x)->foo(f, x, x′s[n]), fs[:, n], xs))
    end
    return Y
end


Base.zero(A::AbstractArray{<:AbstractArray}) = zero.(A)


"""
    BlockMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct BlockMean{Tμ<:AbstractVector{<:MeanFunction}} <: MeanFunction
    μ::Tμ
end
BlockMean(μs::Vararg{<:MeanFunction}) = BlockMean([μs...])
function _map(m::BlockMean, x::BlockData)
    return BlockVector([map(μ, blk) for (μ, blk) in zip(m.μ, blocks(x))])
end
# _map(μ::BlockMean, x::BlockData) = unary_make_vec(map, μ.μ, blocks(x))


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
    items = zip(diag(k.ks), blocks(x), blocks(x′))
    return BlockVector([map(k, blk, blk′) for (k, blk, blk′) in items])
    # return binary_make_vec(map, diag(k.ks), blocks(x), blocks(x′))
end
function _pw(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    x_items, x′_items = enumerate(blocks(x)), enumerate(blocks(x′))
    return BlockMatrix([pw(k.ks[p, q], x, x′) for (p, x) in x_items, (q, x′) in x′_items])
    # return binary_make_mat(pw, k.ks, blocks(x), blocks(x′))
end
_pw(k::BlockCrossKernel, x::BlockData, x′::AV) = _pw(k, x, BlockData([x′]))
_pw(k::BlockCrossKernel, x::AV, x′::BlockData) = _pw(k, BlockData([x]), x′)


# This whole implementation is a hack to ensure that backprop basically works. This will
# change in the future, in particular the constructor will certainly change.

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
    # return binary_make_vec(map, diag(k.ks), blocks(x), blocks(x′))
end
function _pw(k::BlockKernel, x::BlockData, x′::BlockData)
    x_items, x′_items = enumerate(blocks(x)), enumerate(blocks(x′))
    return BlockMatrix([pw(k.ks[p, q], x, x′) for (p, x) in x_items, (q, x′) in x′_items])
    # return binary_make_mat(pw, k.ks, blocks(x), blocks(x′))
end

# Unary methods.
_map(k::BlockKernel, x::BlockData) = _map(k, x, x)
_pw(k::BlockKernel, x::BlockData) = _pw(k, x, x)
