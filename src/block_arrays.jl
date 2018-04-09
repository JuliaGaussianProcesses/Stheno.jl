import Base: *
import BlockArrays: BlockArray, BlockVector, BlockMatrix, BlockVecOrMat, getblock
import LinearAlgebra: adjoint, transpose, Adjoint, Transpose, chol, UpperTriangular
export BlockVector, BlockMatrix, blocksizes, blocklengths

# Do some character saving.
const BV{T} = BlockVector{T}
const BM{T} = BlockMatrix{T}

# Functionality for lazy `transpose` / `adjoint` of block matrix / vector.
for (u, U) in [(:adjoint, :Adjoint), (:transpose, :Transpose)]
    @eval begin
        getblock(x::$U{T, <:BV{T}} where T, n::Int) = $u(getblock(x.parent, n))
        getblock(X::$U{T, <:BM{T}} where T, p::Int, q::Int) = $u(getblock(X.parent, q, p))
    end
end

"""
    BlockVector(xs::Vector{<:AbstractVector{T}}) where T

Construct a `BlockVector` from a collection of `AbstractVector`s.
"""
function BlockVector(xs::Vector{<:AbstractVector{T}}) where T
    x = BlockVector{T}(uninitialized_blocks, length.(xs))
    for (n, x_) in enumerate(xs)
        setblock!(x, x_, n)
    end
    return x
end

"""
    BlockMatrix(Xs::Matrix{<:AbstractMatrix{T}}) where T

Construct a `BlockMatrix` from a matrix of `AbstractMatrix`s.
"""
function BlockMatrix(Xs::Matrix{<:AbstractMatrix{T}}) where T
    X = BlockMatrix{T}(uninitialized_blocks, size.(Xs[:, 1], Ref(1)), size.(Xs[1, :], Ref(2)))
    for q in 1:nblocks(X, 2), p in 1:nblocks(X, 1)
        setblock!(X, Xs[p, q], p, q)
    end
    return X
end

"""
    BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int)

Construct a block matrix with `P` rows and `Q` columns of blocks.
"""
BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int) = BlockMatrix(reshape(xs, P, Q))

"""
    blocksizes(X::BlockArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. 
"""
function blocksizes(X::BlockMatrix, d::Int)
    if d == 1
        return [blocksize(X, n, 1)[1] for n in 1:nblocks(X, 1)]
    elseif d == 2
        return [blocksize(X, 1, n)[2] for n in 1:nblocks(X, 2)]
    else
        throw(error("Boooooooooo, d ∉ (1, 2)."))
    end
end
blocksizes(X::Union{<:Transpose, <:Adjoint}, d::Int) = blocksizes(X.parent, d == 1 ? 2 : 1)
function blocksizes(x::BlockVector, d)
    d == 1 || throw(error("Booooooooo, d ∉ (1,)."))
    return [blocksize(x, n)[1] for n in 1:nblocks(x, 1)]
end
blocklengths(x::BlockVector) = blocksizes(x, 1)
blocklengths(x::Union{<:Transpose, <:Adjoint}) = blocklengths(x.parent)

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = A.block_sizes[2] == B.block_sizes[1]

"""
    *(A::BlockMatrix, x::BlockVector)

Matrix-vector multiplication between `BlockArray`s. Fails if block are not conformal.
"""
function *(A::Union{<:BM{T}, Transpose{T, BM{T}}, Adjoint{T, BM{T}}}, x::BV{T}) where T
    @assert are_conformal(A, x)
    y = BlockVector{T}(uninitialized_blocks, blocksizes(A, 1))
    P, Q = nblocks(A)
    for p in 1:P
        setblock!(y, getblock(A, p, 1) * getblock(x, 1), p)
        for q in 2:Q
            setblock!(y, getblock(y, p) + getblock(A, p, q) * getblock(x, q), p)
        end
    end
    return y
end

"""
    *(A::BlockMatrix, B::BlockMatrix)

Matrix-matrix multiplication between `BlockArray`s. Fails if blocks are not conformal.
"""
function *(A::BlockMatrix{T}, B::BlockMatrix{T}) where T
    @assert are_conformal(A, B)
    C = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(B, 2))
    P, Q, R = nblocks(A, 1), nblocks(A, 2), nblocks(B, 2)
    for p in 1:P, r in 1:R
        setblock!(C, getblock(A, p, 1) * getblock(B, 1, r), p, r)
        for q in 2:Q
            setblock!(C, getblock(C, p, r) + getblock(A, p, q) * getblock(B, q, r), p, r)
        end
    end
    return C
end

"""
    UtDU(A::Symmetric{T, <:BM{T}}) where T<:Real

Get the `UᵀDU` decomposition of `A` in the form of two `BlockMatrices`.
 
Only works for `A` where `is_block_symmetric(A) == true`. Assumes that we want the
upper triangular version.
"""
function UtDU(Asym::Symmetric{T, <:BM{T}}) where T
    A = Asym.data
    @assert blocksizes(X, 1) == blocksizes(X, 2)
    D = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(A, 1))
    U = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(A, 1))

    for j in 1:nblocks(A, 2)

        # Update diagonal.
        setblock!(D, getblock(A, j, j), j, j)
        for k in 1:j-1
            Dk, Ukj = getblock(D, k, k), getblock(U, k, j)
            setblock!(D, getblock(D, j, j) - Xt_A_X(Dk, Ukj), j, j)
        end

        # Update off-diagonals.
        for i in j+1:nblocks(A, 2)
            setblock!(U, getblock(A, i, j), i, j)
            for k in 1:i-1
                Uki, Dk, Ukj = getblock(U, k, i), getblock(D, k, k), getblock(U, k, j)
                setblock!(U, getblock(U, i, j) - Xt_A_Y(Uki, Dk, Ukj), i, j)
            end
            setblock(U, getblock(D, i, i) \ getblock(U, i, j), i, j)
        end
    end
    return UpperTriangular(U)
end
