# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

import Base: +, *, size, getindex, eltype, copy, ctranspose, transpose, chol,
    UpperTriangular, \, logdet
import BlockArrays: BlockArray, BlockVector, BlockMatrix, BlockVecOrMat, getblock,
    blocksize, setblock!, nblocks
export BlockVector, BlockMatrix, SquareDiagonal, blocksizes, blocklengths

# Do some character saving.
const BV{T} = BlockVector{T}
const BM{T} = BlockMatrix{T}
const ABV{T} = AbstractBlockVector{T}
const ABM{T} = AbstractBlockMatrix{T}
const ABVM{T} = AbstractBlockVecOrMat{T}

# # Functionality for lazy `transpose` / `adjoint` of block matrix / vector.
# for (u, U) in [(:adjoint, :Adjoint), (:transpose, :Transpose)]
#     @eval begin
#         getblock(x::$U{T} where T, n::Int) = $u(getblock(x.parent, n))
#         getblock(X::$U{T} where T, p::Int, q::Int) = $u(getblock(X.parent, q, p))
#         nblocks(X::$U{<:Any, <:ABM}) = reverse(nblocks(X.parent))
#         function nblocks(X::$U{<:Any, <:ABM}, N::Int)
#             if N == 1
#                 return nblocks(X.parent, 2)
#             elseif N == 2
#                 return nblocks(X.parent, 1)
#             else
#                 throw(error("Booooo Matrices only have two dimensions."))
#             end
#         end
#     end
# end

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
    BlockMatrix(Xs::Matrix{<:AbstractVecOrMat{T}}) where T

Construct a `BlockMatrix` from a matrix of `AbstractVecOrMat`s.
"""
function BlockMatrix(Xs::Matrix{<:AbstractVecOrMat{T}}) where T
    X = BlockMatrix{T}(uninitialized_blocks, size.(Xs[:, 1], Ref(1)), size.(Xs[1, :], Ref(2)))
    for q in 1:nblocks(X, 2), p in 1:nblocks(X, 1)
        setblock!(X, Xs[p, q], p, q)
    end
    return X
end
BlockMatrix(Xs::Vector{<:AbstractVecOrMat}) = BlockMatrix(reshape(Xs, length(Xs), 1))

"""
    BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int)

Construct a block matrix with `P` rows and `Q` columns of blocks.
"""
BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int) = BlockMatrix(reshape(xs, P, Q))

"""
    blocksizes(X::AbstractBlockMatrix, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. 
"""
function blocksizes(X::AbstractBlockMatrix, d::Int)
    if d == 1
        return [blocksize(X, n, 1)[1] for n in 1:nblocks(X, 1)]
    elseif d == 2
        return [blocksize(X, 1, n)[2] for n in 1:nblocks(X, 2)]
    else
        throw(error("Boooooooooo, d ∉ (1, 2)."))
    end
end
blocksizes(X::Union{<:Transpose, <:Adjoint}, d::Int) = blocksizes(X.parent, d == 1 ? 2 : 1)
blocksizes(X::UpperTriangular{<:Any, <:AbstractBlockMatrix}, d::Int) = blocksizes(X.data, d)
function blocksizes(x::BlockVector, d)
    d == 1 || throw(error("Booooooooo, d ∉ (1,)."))
    return [blocksize(x, n)[1] for n in 1:nblocks(x, 1)]
end
blocklengths(x::BlockVector) = blocksizes(x, 1)
blocklengths(x::Union{<:Transpose, <:Adjoint}) = blocklengths(x.parent)

# Copying a BlockVector.
function copy(a::BV{T}) where T
    b = BlockVector{T}(uninitialized_blocks, blocksizes(a, 1))
    for p in 1:nblocks(b, 1)
        setblock!(b, copy(getblock(a, p)), p)
    end
    return b
end

function copy(A::BM{T}) where T
    B = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(A, 2))
    for q in 1:nblocks(B, 2), p in 1:nblocks(B, 1)
        setblock!(B, copy(getblock(A, p, q)), p, q)
    end
    return B
end

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = blocksizes(A, 2) == blocksizes(B, 1)

"""
    *(A::BlockMatrix, x::BlockVector)

Matrix-vector multiplication between `BlockArray`s. Fails if block are not conformal.
"""
function *(A::Union{<:ABM{T}, Transpose{T, <:ABM{T}}, Adjoint{T, <:ABM{T}}}, x::ABV{T}) where T
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
function *(A::Union{<:ABM{T}, Transpose{T, <:ABM{T}}, Adjoint{T, <:ABM{T}}}, B::Union{<:ABM{T}, Transpose{T, <:ABM{T}}, Adjoint{T, <:ABM{T}}}) where T
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
    SquareDiagonal{T, V<:AM{T}} <: AbstractBlockMatrix{T}

A `SquareDiagonal` is endowed with a stronger form of symmetry than usual for a
`Symmetric`: we require that each block on the diagonal of the `BlockMatrix` that it
represents be `SquareDiagonal`. This is satisfied trivially by non-block matrices, thus
`Symmetric` matrices wrapping a `Matrix` are also `BlockSymmetric`. If a block on the
diagonal of a `SquareDiagonal` matrix is itself a `BlockMatrix`, then  we require that it
also be `SquareDiagonal`.
"""
struct SquareDiagonal{T, TX<:ABM{T}} <: AbstractBlockMatrix{T}
    X::TX
    function SquareDiagonal(X::BlockMatrix)
        @assert blocksizes(X, 1) == blocksizes(X, 2)
        return new{eltype(X), typeof(X)}(X)
    end
end
const SD{T, TX} = SquareDiagonal{T, TX}
nblocks(X::SquareDiagonal) = nblocks(X.X)
nblocks(X::SquareDiagonal, i::Int) = nblocks(X.X, i)
blocksize(X::SquareDiagonal, N::Int...) = blocksize(X.X, N...)
getblock(X::SquareDiagonal, p::Int...) = getblock(X.X, p...)
setblock!(X::SquareDiagonal, v, p::Int...) = setblock!(X.X, v, p...)
size(X::SquareDiagonal) = size(X.X)
getindex(X::SquareDiagonal, p::Int, q::Int) = getindex(X.X, (p < q ? (p, q) : (q, p))...)
eltype(X::SquareDiagonal) = eltype(X.X)
copy(X::SquareDiagonal{T}) where T = SquareDiagonal(copy(X.X))

"""
    getblock(X::UpperTriangular{T, <:SquareDiagonal{T}} where T, p::Int, q::Int)

Return block of zeros of the appropriate size if p > q.
"""
getblock(X::UpperTriangular{T}, p::Int, q::Int) where T =
    p > q ? zeros(T, blocksize(X.data, p, q)) : getblock(X.data, p, q)

"""
    chol(A::SquareDiagonal{T, <:BM{T}}) where T<:Real

Get the Cholesky decomposition of `A` in the form of a `BlockMatrix`.

Only works for `A` where `is_block_symmetric(A) == true`. Assumes that we want the
upper triangular version.
"""
function chol(A::SquareDiagonal{T, <:BM{T}}) where T<:Real
    U = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(A, 1))
    for j in 1:nblocks(A, 2)

        # Update off-diagonals.
        for i in 1:j-1
            setblock!(U, getblock(A, i, j), i, j)
            for k in 1:i-1
                Uki, Ukj = getblock(U, k, i), getblock(U, k, j)
                setblock!(U, getblock(U, i, j) - Uki' * Ukj, i, j)
            end
            setblock!(U, getblock(U, i, i)' \ getblock(U, i, j), i, j)
        end

        # Update diagonal.
        setblock!(U, getblock(A, j, j), j, j)
        for k in 1:j-1
            Ukk, Ukj = getblock(U, k, k), getblock(U, k, j)
            setblock!(U, getblock(U, j, j) - Ukj' * Ukj, j, j)
        end
        setblock!(U, chol(getblock(U, j, j)), j, j)
    end
    return UpperTriangular(SquareDiagonal(U))
end

function \(U::UpperTriangular{T, <:SD}, x::ABV{T}) where T<:Real
    y = BlockVector{T}(uninitialized_blocks, blocksizes(U, 1))
    for p in reverse(1:nblocks(y, 1))
        setblock!(y, getblock(x, p), p)
        for p′ in p+1:nblocks(y, 1)
            setblock!(y, getblock(y, p) - getblock(U, p, p′) * getblock(y, p′), p)
        end
        setblock!(y, getblock(U, p, p) \ getblock(y, p), p)
    end
    return y
end

function \(U::UpperTriangular{T, <:SD{T}}, X::ABM{T}) where T<:Real
    Y = BlockMatrix{T}(uninitialized_blocks, blocksizes(U, 1), blocksizes(X, 2))
    for q in 1:nblocks(Y, 2), p in reverse(1:nblocks(Y, 1))
        setblock!(Y, getblock(X, p, q), p, q)
        for p′ in p+1:nblocks(Y, 1)
            setblock!(Y, getblock(Y, p, q) - getblock(U, p, p′) * getblock(Y, p′, q), p, q)
        end
        setblock!(Y, getblock(U, p, p) \ getblock(Y, p, q), p, q)
    end
    return Y
end

function \(L::Adjoint{T, <:UpperTriangular{T, <:SD}}, x::ABV{T}) where T<:Real
    y = BlockVector{T}(uninitialized_blocks, blocksizes(L, 1))
    for p in 1:nblocks(y, 1)
        setblock!(y, getblock(x, p), p)
        for p′ in 1:p-1
            setblock!(y, getblock(y, p) - getblock(L, p, p′) * getblock(y, p′), p)
        end
        setblock!(y, getblock(L, p, p) \ getblock(y, p), p)
    end
    return y
end

function \(L::Adjoint{T, <:UpperTriangular{T, <:SD}}, X::ABM{T}) where T<:Real
    Y = BlockMatrix{T}(uninitialized_blocks, blocksizes(L, 1), blocksizes(X, 2))
    for q in 1:nblocks(Y, 2), p in 1:nblocks(Y, 1)
        setblock!(Y, getblock(X, p, q), p, q)
        for p′ in 1:p-1
            setblock!(Y, getblock(Y, p, q) - getblock(L, p, p′) * getblock(Y, p′, q), p, q)
        end
        setblock!(Y, getblock(L, p, p) \ getblock(Y, p, q), p, q)
    end
    return Y
end

\(L::Transpose{T, <:UpperTriangular{T, <:SD}}, X::ABVM{T}) where T<:Real =
    adjoint(L.parent) \ X

import LinearAlgebra: UniformScaling
function +(u::UniformScaling, X::SquareDiagonal)
    Y = copy(X)
    for p in 1:nblocks(Y, 1)
        setblock!(Y, getblock(Y, p, p) + u, p, p)
    end
    return Y
end
function +(X::SquareDiagonal, u::UniformScaling)
    Y = copy(X)
    for p in 1:nblocks(Y, 1)
        setblock!(Y, u + getblock(Y, p, p), p, p)
    end
    return Y
end

# Define addition and subtraction for compatible block matrices and vectors.
import Base: +, -
for foo in [:+, :-]
    @eval function $foo(A::BV{T}, B::BV{T}) where T
        @assert blocksizes(A, 1) == blocksizes(B, 1)
        C = BlockVector{T}(uninitialized_blocks, blocksizes(A, 1))
        for p in 1:nblocks(C, 1)
            setblock!(C, $foo(getblock(A, p), getblock(B, p)), p)
        end
        return C
    end
    @eval function $foo(A::BM{T}, B::BM{T}) where T
        @assert blocksizes(A, 1) == blocksizes(B, 1)
        @assert blocksizes(A, 2) == blocksizes(B, 2)
        C = BlockMatrix{T}(uninitialized_blocks, blocksizes(A, 1), blocksizes(A, 2))
        for q in 1:nblocks(C, 2), p in 1:nblocks(C, 1)
            setblock!(C, $foo(getblock(A, p, q), getblock(B, p, q)), p, q)
        end
        return C
    end
end
