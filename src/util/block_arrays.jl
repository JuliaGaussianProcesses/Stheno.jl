# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

using FillArrays, LinearAlgebra, BlockArrays
using FillArrays: Fill

import Base: +, *, size, getindex, eltype, copy, \, vec, getproperty
import LinearAlgebra: UpperTriangular, LowerTriangular, logdet, Symmetric, transpose,
    adjoint, AdjOrTrans, AdjOrTransAbsMat, cholesky
import BlockArrays: BlockArray, BlockVector, BlockMatrix, BlockVecOrMat, getblock,
    blocksize, setblock!, nblocks

# Do some character saving.
const BV{T} = BlockVector{T}
const BM{T} = BlockMatrix{T}
const ABV{T} = AbstractBlockVector{T}
const ABM{T} = AbstractBlockMatrix{T}
const ABVM{T} = AbstractBlockVecOrMat{T}
const LUABM{T} = Union{ABM, LowerTriangular{T, <:ABM}, UpperTriangular{T}, <:ABM}

const AdjOrTransABM{T} = Union{ABM{T}, AdjOrTrans{T, <:ABM{T}}}
const AdjTransTriABM{T} = Union{AbstractTriangular{T, <:ABM{T}}, AdjOrTransABM{T}}
const TriABM{T} = Union{AbstractTriangular{T, <:ABM{T}}, ABM{T}}
const BlockTri{T} = AbstractTriangular{T, <:ABM{T}}
const BlockAdjOrTrans{T} = AdjOrTrans{T, <:ABM{T}}

unbox(X::AbstractBlockMatrix) = X
unbox(X::Symmetric) = unbox(X.data)
unbox(X::AbstractMatrix) = X



####################################### Various util #######################################

import BlockArrays: BlockVector, BlockMatrix
export BlockVector, BlockMatrix, blocksizes, blocklengths

"""
    BlockVector(xs::Vector{<:AbstractVector{T}}) where T

Construct a `BlockVector` from a collection of `AbstractVector`s.
"""
function BlockArrays.BlockVector(xs::Vector{<:AbstractVector{T}}) where T
    x = BlockVector{T}(undef_blocks, length.(xs))
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
    X = BlockMatrix{T}(undef_blocks, size.(Xs[:, 1], Ref(1)), size.(Xs[1, :], Ref(2)))
    for q in 1:nblocks(X, 2), p in 1:nblocks(X, 1)
        setblock!(X, Xs[p, q], p, q)
    end
    return X
end
BlockMatrix(Xs::Vector{<:AbstractVecOrMat}) = BlockMatrix(reshape(Xs, length(Xs), 1))
BlockMatrix(x::AbstractVecOrMat) = BlockMatrix([x])

"""
    BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int)

Construct a block matrix with `P` rows and `Q` columns of blocks.
"""
BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int) = BlockMatrix(reshape(xs, P, Q))

"""
    blocksizes(X::AbstractBlockArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. 
"""
function blocksizes(X::AbstractBlockArray, d::Int)
    @assert d > 0 && d <= ndims(X)
    idxs = [1 for _ in 1:length(size(X))]
    block_sizes = Vector{Int}(undef, nblocks(X, d))
    for p in eachindex(block_sizes)
        idxs[d] = p
        block_sizes[p] = blocksize(X, idxs...)[d]
    end
    return block_sizes
end
blocksizes(X::AbstractBlockArray) = ([blocksizes(X, d) for d in 1:length(size(X))]...,)
blocklengths(x::BlockVector) = blocksizes(x, 1)



################################# Symmetric BlockMatrices ##############################

const BS{T} = Symmetric{T, <:AbstractBlockMatrix{T}}
unbox(X::BS) = X.data
nblocks(X::BS) = nblocks(unbox(X))
nblocks(X::BS, i::Int) = nblocks(unbox(X), i)
blocksize(X::BS, N::Int...) = blocksize(unbox(X), N...)
blocksizes(X::BS, d::Int...) = blocksizes(unbox(X), d...)

function getblock(X::BS, p::Int, q::Int)
    @assert blocksizes(X, 1) == blocksizes(X, 2)
    X_, uplo = unbox(X), X.uplo
    if p < q
        return uplo == 'U' ? getblock(X_, p, q) : transpose(getblock(X_, q, p))
    elseif p == q
        return Symmetric(getblock(X_, p, q))
    else
        return uplo == 'U' ? transpose(getblock(X_, q, p)) : getblock(X_, p, q)
    end
end



######################## Util for triangular block matrices ######################

@inline unbox(U::UpperTriangular{T, <:ABM{T}} where T) = U.data
function blocksize(U::UpperTriangular{T, <:ABM{T}} where T, p::Int, q::Int)
    return blocksize(unbox(U), p, q)
end
blocksizes(U::UpperTriangular{T, <:ABM{T}} where T, d::Int) = blocksizes(unbox(U), d)
blocksizes(U::UpperTriangular{T, <:ABM{T}} where T) = blocksizes(unbox(U))
function getblock(U::UpperTriangular{T, <:ABM{T}}, p::Int, q::Int) where T
    @assert blocksizes(U, 1) == blocksizes(U, 2)
    if p > q
        return Zeros{T}(blocksize(U, p, q)...)
    elseif p == q
        return UpperTriangular(getblock(unbox(U), p, q))
    else
        return getblock(unbox(U), p, q)
    end
end
nblocks(U::UpperTriangular{T, <:ABM{T}} where T, d::Int...) = nblocks(unbox(U), d...)
function BlockMatrix(U::UpperTriangular{T, <:ABM{T}}) where T
    B = BlockMatrix{T}(undef_blocks, blocksizes(U)...)
    for q in 1:nblocks(U, 2)
        for p in 1:q-1
            setblock!(B, getblock(U, p, q), p, q)
        end
        setblock!(B, UpperTriangular(getblock(U, q, q)), q, q)
        for p in q+1:nblocks(U, 1)
            setblock!(B, Zeros{T}(blocksize(U, p, q)), p, q)
        end
    end
    return B
end

@inline unbox(L::LowerTriangular{T, <:ABM{T}} where T) = L.data
function blocksize(L::LowerTriangular{T, <:ABM{T}} where T, p::Int, q::Int)
    return blocksize(unbox(L), p, q)
end
blocksizes(L::LowerTriangular{T, <:ABM{T}} where T, d::Int) = blocksizes(unbox(L), d)
blocksizes(L::LowerTriangular{T, <:ABM{T}} where T) = blocksizes(unbox(L))
function getblock(L::LowerTriangular{T, <:ABM{T}}, p::Int, q::Int) where T
    @assert blocksizes(L, 1) == blocksizes(L, 2)
    if p > q
        return getblock(unbox(L), p, q)
    elseif p == q
        return LowerTriangular(getblock(unbox(L), p, q))
    else
        return Zeros{T}(blocksize(L, p, q)...)
    end
end
nblocks(L::LowerTriangular{T, <:ABM{T}} where T, d::Int...) = nblocks(unbox(L), d...)
function BlockMatrix(L::LowerTriangular{T, <:ABM{T}}) where T
    B = BlockMatrix{T}(undef_blocks, blocksizes(L)...)
    for q in 1:nblocks(L, 2)
        for p in 1:q-1
            setblock!(B, Zeros{T}(blocksize(L, p, q)), p, q)
        end
        setblock!(B, LowerTriangular(getblock(L, q, q)), q, q)
        for p in q+1:nblocks(L, 1)
            setblock!(B, getblock(L, p, q), p, q)
        end
    end
    return B
end



####################################### Copying ######################################

function copy(a::BlockVector{T}) where T
    b = BlockVector{T}(undef_blocks, blocksizes(a, 1))
    for p in 1:nblocks(b, 1)
        setblock!(b, copy(getblock(a, p)), p)
    end
    return b
end

function copy(A::BlockMatrix{T, V}) where {T, V<:AbstractMatrix{T}}
    B = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 2))
    for q in 1:nblocks(B, 2), p in 1:nblocks(B, 1)
        setblock!(B, copy(getblock(A, p, q)), p, q)
    end
    return B
end

copy(B::BS{T, <:ABM{T}} where T) = Symmetric(copy(unbox(B)))
copy(L::LowerTriangular{T, <:BS{T}} where T) = LowerTriangular(copy(unbox(L)))
copy(U::UpperTriangular{T, <:BS{T}} where T) = UpperTriangular(copy(unbox(U)))



####################################### Transposition ######################################

nblocks(A::AdjOrTrans{T, <:TriABM{T}} where T) = reverse(nblocks(A'))
function nblocks(A::AdjOrTrans{T, <:TriABM{T}} where T, d::Int) where {N}
    @assert d ∈ (1, 2)
    return nblocks(A)[d]
end
function blocksizes(A::AdjOrTrans{T, <:TriABM{T}} where T, d::Int)
    @assert d ∈ (1, 2)
    return blocksizes(A', d == 1 ? 2 : 1)
end
function getblock(A::Adjoint{T, <:TriABM{T}} where T<:Real, p::Int, q::Int)
    return getblock(A.parent, q, p)'
end
function getblock(A::Transpose{T, <:TriABM{T}} where T<:Real, p::Int, q::Int)
    return transpose(getblock(A.parent, q, p))
end



####################################### Multiplication #####################################

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = blocksizes(A, 2) == blocksizes(B, 1)

"""
    *(A::AdjTransTriABM, x::BlockVector)

Matrix-vector multiplication between `BlockArray`s. Fails if block are not conformal.
"""
function *(
    A::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
    x::ABV{T},
) where T
    @assert are_conformal(A, x)
    y = BlockVector{T}(undef_blocks, blocksizes(A, 1))
    P, Q = nblocks(A)
    for p in 1:P
        setblock!(y, getblock(A, p, 1) * getblock(x, 1), p)
        for q in 2:Q
            setblock!(y, getblock(y, p) + getblock(A, p, q) * getblock(x, q), p)
        end
    end
    return y
end
function *(A::AdjTransTriABM{T}, b::AV{T}) where {T}
    @assert nblocks(A, 2) == 1
    return A * BlockVector([b])
end
function *(A::AdjTransTriABM{T}, b::FillArrays.Zeros{T,1}) where T
    @show size(A), size(b)
    return invoke(*, Tuple{AbstractMatrix, typeof(b)}, A, b)
end



"""
    *(A::AdjTransTriABM{T}, B::AdjTransTriABM{T}) where T

Matrix-matrix multiplication between `BlockArray`s. Fails if blocks are not conformal.
"""
function *(
    A::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
    B::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
) where T
    @assert are_conformal(A, B)
    C = BlockMatrix{T}(undef_blocks, blocksizes(A, 1), blocksizes(B, 2))
    P, Q, R = nblocks(A, 1), nblocks(A, 2), nblocks(B, 2)
    for p in 1:P, r in 1:R
        setblock!(C, getblock(A, p, 1) * getblock(B, 1, r), p, r)
        for q in 2:Q
            setblock!(C, getblock(C, p, r) + getblock(A, p, q) * getblock(B, q, r), p, r)
        end
    end
    return C
end
*(A::AdjTransTriABM{T}, B::AM{T}) where T = A * BlockMatrix([B])
*(A::AM{T}, B::AdjTransTriABM{T}) where T = BlockMatrix([A]) * B

function *(A::LazyPDMat{T, <:Symmetric{T, <:ABM{T}}}, B::AdjTransTriABM{T}) where T
    return unbox(unbox(A)) * B
end

const UpperOrAdjLower{T} = Union{
    UpperTriangular{T, <:ABM{T}},
    AdjOrTrans{T, <:LowerTriangular{T, <:ABM{T}}},
}
const LowerOrAdjUpper{T} = Union{
    LowerTriangular{T, <:ABM{T}},
    AdjOrTrans{T, <:UpperTriangular{T, <:ABM{T}}},
}

function \(U::UpperOrAdjLower{T}, x::ABV{T}) where T<:Real
    @assert are_conformal(unbox(U), x)
    y = BlockVector{T}(undef_blocks, blocksizes(U, 1))
    for p in reverse(1:nblocks(y, 1))
        setblock!(y, getblock(x, p), p)
        for p′ in p+1:nblocks(y, 1)
            setblock!(y, getblock(y, p) - getblock(U, p, p′) * getblock(y, p′), p)
        end
        setblock!(y, getblock(U, p, p) \ getblock(y, p), p)
    end
    return y
end

function _block_ldiv_mat_upper(U, X)
    @assert are_conformal(unbox(U), X)
    Y = BlockMatrix{eltype(U)}(undef_blocks, blocksizes(U, 1), blocksizes(X, 2))
    for q in 1:nblocks(Y, 2), p in reverse(1:nblocks(Y, 1))
        setblock!(Y, getblock(X, p, q), p, q)
        for p′ in p+1:nblocks(Y, 1)
            setblock!(Y, getblock(Y, p, q) - getblock(U, p, p′) * getblock(Y, p′, q), p, q)
        end
        setblock!(Y, getblock(U, p, p) \ getblock(Y, p, q), p, q)
    end
    return Y
end
function \(U::UpperOrAdjLower{T}, X::Union{ABM{T}, BlockTri{T}}) where T<:Real
    return _block_ldiv_mat_upper(U, X)
end
function \(U::UpperOrAdjLower{T}, X::Adjoint{T, <:UpperTriangular{T, <:ABM{T}}}) where T<:Real
    return _block_ldiv_mat_upper(U, X)
end

function \(L::LowerOrAdjUpper{T}, x::ABV{T}) where T<:Real
    @assert are_conformal(unbox(L), x)
    y = BlockVector{T}(undef_blocks, blocksizes(L, 1))
    for p in 1:nblocks(y, 1)
        setblock!(y, getblock(x, p), p)
        for p′ in 1:p-1
            setblock!(y, getblock(y, p) - getblock(L, p, p′) * getblock(y, p′), p)
        end
        setblock!(y, getblock(L, p, p) \ getblock(y, p), p)
    end
    return y
end

function _block_ldiv_mat_lower(L, X)
    @assert are_conformal(unbox(L), X)
    Y = BlockMatrix{eltype(L)}(undef_blocks, blocksizes(L, 1), blocksizes(X, 2))
    for q in 1:nblocks(Y, 2), p in 1:nblocks(Y, 1)
        setblock!(Y, getblock(X, p, q), p, q)
        for p′ in 1:p-1
            setblock!(Y, getblock(Y, p, q) - getblock(L, p, p′) * getblock(Y, p′, q), p, q)
        end
        setblock!(Y, getblock(L, p, p) \ getblock(Y, p, q), p, q)
    end
    return Y
end
function \(L::LowerOrAdjUpper{T}, X::Union{ABM{T}, BlockTri{T}}) where T<:Real
    return _block_ldiv_mat_lower(L, X)
end
function \(L::LowerOrAdjUpper{T}, X::Adjoint{T, <:UpperTriangular{T, <:ABM{T}}}) where T<:Real
    return _block_ldiv_mat_lower(L, X)
end


"""
    _chol(A::Symmetric{T, <:BlockMatrix{T}}) where T<:Real

Get the Cholesky decomposition of `A` in the form of a `BlockMatrix`.

Only works for `A` where `is_block_symmetric(A) == true`. Assumes that we want the
upper triangular version.
"""
function _chol(A::Symmetric{T, <:BlockMatrix{T, V}}) where {T<:Real, V<:AbstractMatrix{T}}

    U = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 1))

    # Do an initial pass to fill each of the blocks with Zeros. This is cheap
    for q in 1:nblocks(U, 2), p in 1:nblocks(U, 1)
        setblock!(U, Zeros{T}(blocksize(A, p, q)...), p, q)
    end

    # Fill out the upper triangle with the Cholesky
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
        setblock!(U, cholesky(getblock(U, j, j)).U, j, j)
    end
    return U
end

# Get only implement one of these methods for now.
function cholesky(A::Symmetric{T, <:BlockMatrix{T, <:AbstractMatrix{T}}} where T<:Real)
    return Cholesky(_chol(A), :U, 0)
end

logdet(A::Cholesky{T, <:AbstractBlockMatrix{T}}) where T = 2 * logdet(A.U)
function logdet(U::UpperTriangular{T, <:AbstractBlockMatrix{T}}) where T
    val = zero(T)
    for n in 1:nblocks(U, 1)
        val += logdet(UpperTriangular(getblock(unbox(U), n, n)))
    end
    return val
end

function +(u::UniformScaling, X::AbstractBlockMatrix)
    @assert blocksizes(X, 1) == blocksizes(X, 2)
    Y = copy(X)
    for p in 1:nblocks(Y, 1)
        setblock!(Y, getblock(Y, p, p) + u, p, p)
    end
    return Y
end
+(u::UniformScaling, X::Symmetric{T, <:ABM{T}} where T) = Symmetric(u + unbox(X))
function +(X::AbstractBlockMatrix, u::UniformScaling)
    @assert blocksizes(X, 1) == blocksizes(X, 2)
    Y = copy(X)
    for p in 1:nblocks(Y, 1)
        setblock!(Y, u + getblock(Y, p, p), p, p)
    end
    return Y
end
+(X::Symmetric{T, <:ABM{T}} where T, u::UniformScaling) = Symmetric(unbox(X) + u)

# Define addition and subtraction for compatible block matrices and vectors.
import Base: +, -
for foo in [:+, :-]
    @eval function $foo(A::BV{T}, B::BV{T}) where T
        @assert blocksizes(A, 1) == blocksizes(B, 1)
        C = BlockVector{T}(undef_blocks, blocksizes(A, 1))
        for p in 1:nblocks(C, 1)
            setblock!(C, $foo(getblock(A, p), getblock(B, p)), p)
        end
        return C
    end
    @eval function $foo(A::BM{T}, B::BM{T}) where T
        @assert blocksizes(A, 1) == blocksizes(B, 1)
        @assert blocksizes(A, 2) == blocksizes(B, 2)
        C = BlockMatrix{T}(undef_blocks, blocksizes(A, 1), blocksizes(A, 2))
        for q in 1:nblocks(C, 2), p in 1:nblocks(C, 1)
            setblock!(C, $foo(getblock(A, p, q), getblock(B, p, q)), p, q)
        end
        return C
    end
end



#################################### Broadcasting ##########################################
# Override the usual broadcasting machinery for BlockArrays. This is a pretty disgusting
# hack. At the time of writing it, I was also writing my first year report, so was short of
# time. This is generally an open problem for AbstractBlockArrays which really does need to
# be resolved at some point.

# Very specific `broadcast` method for particular use case. Needs to be generalised.
function broadcast(f, A::AbstractBlockVector, b::Real)
    return BlockVector([broadcast(f, getblock(A, p), b) for p in 1:nblocks(A, 1)])
end
function broadcast(f, A::AbstractBlockMatrix, b::Real)
    return BlockMatrix([broadcast(f, getblock(A, p, q), b)
        for p in 1:nblocks(A, 1), q in 1:nblocks(A, 2)])
end
