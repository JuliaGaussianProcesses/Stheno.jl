# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

using BlockArrays: cumulsizes, blocksizes, blocksizes, BlockSizes

import Base: +, *, size, getindex, eltype, copy, \, vec, getproperty, zero
import LinearAlgebra: UpperTriangular, LowerTriangular, logdet, Symmetric, transpose,
    adjoint, AdjOrTrans, AdjOrTransAbsMat, cholesky!, logdet, ldiv!, mul!, logabsdet
import BlockArrays: BlockArray, BlockVector, BlockMatrix, BlockVecOrMat, getblock,
    blocksize, setblock!, nblocks, getblock!

# Do some character saving.
const BV{T} = BlockVector{T}
const BM{T} = BlockMatrix{T}
const ABV{T} = AbstractBlockVector{T}
const ABM{T} = AbstractBlockMatrix{T}

const BlockCholesky{T, V} = Cholesky{T, <:BlockMatrix{T, V}}

const BlockArrayAdjoint = NamedTuple{(:blocks, :block_sizes)}


####################################### Various util #######################################

import BlockArrays: BlockVector, BlockMatrix, blocksizes, cumulsizes, AbstractBlockSizes
export BlockVector, BlockMatrix, blocksizes, blocklengths

@adjoint function _BlockArray(
    blocks::R,
    block_sizes::BS,
) where {T, N, R<:AbstractArray{<:AbstractArray{T,N},N}, BS<:AbstractBlockSizes{N}}
    back(Δ::NamedTuple{(:blocks, :block_sizes)}) = (Δ.blocks, nothing)
    back(Δ::BlockArray) = (Δ.blocks, nothing)
    back(Δ::AbstractArray) = (BlockArray(Δ, block_sizes).blocks, nothing)
    return _BlockArray(blocks, block_sizes), back
end

@adjoint Vector(x::BlockVector) = Vector(x), Δ::Vector->(BlockArray(Δ, blocksizes(x)),)
@adjoint function Matrix(X::BlockMatrix)
    back(Δ::Matrix) = (BlockArray(Δ, blocksizes(X)),)
    back(Δ::UpperTriangular{T, <:Matrix{T}} where {T}) = back(Matrix(Δ))
    return Matrix(X), back
end

# Tell Zygote to ignore some stuff that doesn't involve gradient info.
_get_block_sizes(xs::Vector) = (length.(xs),)
_get_block_sizes(Xs) = (size.(Xs[:, 1], 1), size.(Xs[1, :], 2))
@nograd _get_block_sizes

"""
    blocksizes(X::AbstractArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. 
"""
blocksizes(X::AbstractArray, d::Int) = diff(cumulsizes(X, d))
@nograd blocksizes

"""
    BlockVector(xs::AbstractVector{<:AbstractVector})

Construct a `BlockVector` from a collection of `AbstractVector`s.
"""
BlockVector(xs::AV{<:AV}) = _BlockArray(xs, _get_block_sizes(xs)...)

"""
    BlockMatrix(Xs::Matrix{<:AbstractVecOrMat})

Construct a `BlockMatrix` from a matrix of `AbstractVecOrMat`s.
"""
BlockMatrix(Xs::Matrix{<:AbstractVecOrMat}) = _BlockArray(Xs, _get_block_sizes(Xs)...)

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = cumulsizes(A, 2) == cumulsizes(B, 1)



# BlockMatrix(Xs::Vector{<:AbstractVecOrMat}) = BlockMatrix(reshape(Xs, length(Xs), 1))
# BlockMatrix(x::AbstractVecOrMat) = BlockMatrix([x])

# """
#     BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int)

# Construct a block matrix with `P` rows and `Q` columns of blocks.
# """
# BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int) = BlockMatrix(reshape(xs, P, Q))

# zero(x::AbstractBlockVector) = BlockVector([zero(getblock(x, n)) for n in 1:nblocks(x, 1)])
# function zero(X::AbstractBlockMatrix)
#     blocks = [zero(getblock(X, p, q)) for p in 1:nblocks(X, 1), q in 1:nblocks(X, 2)]
#     return BlockMatrix(blocks)
# end

# # Convert from a BlockVector to a single-column BlockMatrix
# function _to_block_matrix(x::BlockVector)
#     return _BlockArray(reshape(reshape.(x.blocks, :, 1), :, 1), blocksizes(x, 1), [1])
# end
# @adjoint function _to_block_matrix(x::BlockVector)
#     X = _to_block_matrix(x)
#     back(Δ::BlockMatrix) = (_to_block_vector(Δ),)
#     back(Δ::AbstractMatrix) = back(BlockArray(Δ), blocksizes(X)...)
#     return X, back
# end

# # Convert a single-column BlockMatrix to a BlockVector.
# function _to_block_vector(X::BlockMatrix)
#     @assert size(X, 2) == 1
#     return _BlockArray(reshape(reshape.(X.blocks, :), :), blocksizes(X, 1))
# end
# @adjoint function _to_block_vector(X::BlockMatrix)
#     x = _to_block_vector(X)
#     back(Δ::BlockVector) = (_to_block_matrix(Δ),)
#     back(Δ::AbstractVector) = back(BlockArray(Δ), blocksizes(x)...)
#     return x, back
# end



# #
# # Dense BlockArrays
# #

# *(a::BlockVector, b::BlockMatrix) = _to_block_matrix(a) * b

# function *(A::BlockMatrix{T}, x::BlockVector{V}) where {T, V}
#     Ps = blocksizes(A, 1)
#     y = _BlockArray([Vector{promote_type(T, V)}(undef, P) for P in Ps], Ps)
#     return mul!(y, A, x)
# end

# @adjoint *(A::BlockMatrix, x::BlockVector) = A * x, Δ::BlockVector -> (Δ * x', A' * Δ)

# function mul!(y::BlockVector, A::BlockMatrix, x::BlockVector)
#     @assert are_conformal(A, x) && are_conformal(A', y)
#     for r in 1:nblocks(A, 1)
#         mul!(y[Block(r)], A[Block(r, 1)], x[Block(1)])
#         for c in 2:nblocks(A, 2)
#             y[Block(r)] = y[Block(r)] + A[Block(r, c)] * x[Block(c)]
#         end
#     end
#     return y
# end

# function *(A::BlockMatrix{T}, B::BlockMatrix{V}) where {T, V}
#     Ps, Qs = blocksizes(A, 1), blocksizes(B, 2)
#     C = _BlockArray([Matrix{promote_type(T, V)}(undef, P, Q) for P in Ps, Q in Qs], Ps, Qs)
#     return mul!(C, A, B)
# end

# @adjoint *(A::BlockMatrix, B::BlockMatrix) = A * B, Δ::BlockMatrix->(Δ * B', A' * Δ)

# function mul!(C::BlockMatrix, A::BlockMatrix, B::BlockMatrix)
#     @assert are_conformal(A, B)
#     @assert cumulsizes(A, 1) == cumulsizes(C, 1)
#     @assert cumulsizes(C, 2) == cumulsizes(B, 2)

#     P, Q, R = nblocks(A, 1), nblocks(A, 2), nblocks(B, 2)
#     for p in 1:P, r in 1:R
#         mul!(C[Block(p, r)], A[Block(p, 1)], B[Block(1, r)])
#         for q in 2:Q
#             C[Block(p, r)] = C[Block(p, r)] + A[Block(p, q)] * B[Block(q, r)]
#         end
#     end
#     return C
# end

# function cholesky_and_adjoint(A::BlockMatrix{T, V}) where {T, V}
#     U = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 2))
#     D = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 2))
#     backs = Vector{Any}(undef, nblocks(A, 2))

#     # Do an initial pass to fill each of the blocks with zeros. This is cheap.
#     for q in 1:nblocks(U, 2), p in 1:nblocks(U, 1)
#         U[Block(p, q)] = zeros{T}(blocksize(A, (p, q))...)
#     end

#     # Fill out the upper triangle with the Cholesky.
#     for j in 1:nblocks(A, 2)

#         # Update off-diagonals.
#         for i in 1:j-1
#             # U[Block(i, j)] = copy(A[Block(i, j)]
#             D[Block(i, j)] = copy(A[Block(i, j)])
#             for k in 1:i-1
#                 D[Block(i, j)] -= U[Block(k, i)]' * U[Block(k, j)]
#             end
#             U[Block(i, j)] = U[Block(i, i)]' \ D[Block(i, j)]
#             # ldiv!(UpperTriangular(U[Block(i, i)])', U[Block(i, j)])
#         end

#         # Update diagonal.
#         D[Block(j, j)] = copy(A[Block(j, j)])
#         for k in 1:j-1
#             D[Block(j, j)] -= U[Block(k, j)]' * U[Block(k, j)]
#         end
#         U_tmp, back = Zygote.forward(A->cholesky(A).U, D[Block(j, j)])
#         U[Block(j, j)] = U_tmp
#         backs[j] = back
#         # U[Block(j, j)] = cholesky(U[Block(j, j)]).U
#     end

#     _to_block(Ū::AbstractMatrix) = BlockArray(copy(Ū), blocksizes(A))
#     _to_block(Ū::AbstractBlockMatrix) = copy(Ū)
#     function _to_block(Ū::BlockUpperTriangular)
#         return copy(Ū.data)
#     end

#     return Cholesky(U, :U, 0), function(Δ::NamedTuple)

#         Ū = _to_block(Δ.factors)
#         Ā = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 1))

#         for j in reverse(1:nblocks(A, 2))

#             Ā[Block(j, j)] = first(backs[j](Ū[Block(j, j)]))
#             for k in reverse(1:j-1)
#                 Ū[Block(k, j)] -= U[Block(k, j)] * (Ā[Block(j, j)] + Ā[Block(j, j)]')
#             end

#             for i in reverse(1:j-1)
#                 Ā[Block(i, j)] = U[Block(i, i)] \ Ū[Block(i, j)]
#                 Ū[Block(i, i)] -= U[Block(i, j)] * Ā[Block(i, j)]'
#                 for k in reverse(1:i-1)
#                     Ū[Block(k, i)] -= U[Block(k, j)] * Ā[Block(i, j)]'
#                     Ū[Block(k, j)] -= U[Block(k, i)] * Ā[Block(i, j)]
#                 end
#             end
#         end

#         # Zero-out the lower triangle.
#         for q in 1:nblocks(A, 1)
#             for p in q+1:nblocks(A, 2)
#                 Ā[Block(p, q)] = zeros(size(A[Block(p, q)]))
#             end
#         end
#         return (Ā,)
#     end
# end

# cholesky(A::BlockMatrix{<:Real}) = first(cholesky_and_adjoint(A))
# @adjoint cholesky(A::BlockMatrix{<:Real}) = cholesky_and_adjoint(A)

# cholesky(A::Symmetric{T, <:BlockMatrix{T}} where {T<:Real}) = cholesky(A.data)

# # A slightly strange util function that shouldn't ever be used outside of `logdet`.
# reduce_diag(f, A::Matrix{<:Real}) = sum(f, view(A, diagind(A)))

# function reduce_diag(f, A::BlockMatrix{<:Real})
#     return sum([reduce_diag(f, getblock(A, n, n)) for n in 1:nblocks(A, 1)])
# end

# logdet(C::BlockCholesky{<:Real}) = 2 * reduce_diag(log, C.factors)

# @adjoint function logdet(C::BlockCholesky{<:Real})
#     return logdet(C), function(Δ::Real)
#         function update_diag!(X::Matrix, A::Matrix)
#             X[diagind(X)] .= (2 * Δ) ./ A[diagind(A)]
#             return X
#         end
#         function update_diag!(X::BlockMatrix, A::BlockMatrix)
#             for n in 1:nblocks(A)[1]
#                 update_diag!(getblock(X, n, n), getblock(A, n, n))
#             end
#             return X
#         end
#         factors = update_diag!(zero(C.factors), C.factors)
#         return ((factors=factors, uplo=nothing, info=nothing),)
#     end
# end

# ldiv!(C::BlockCholesky{<:Real}, x::BlockVector) = ldiv!(C.U, ldiv!(C.U', x))
# ldiv!(C::BlockCholesky{<:Real}, X::BlockMatrix) = ldiv!(C.U, ldiv!(C.U', X))

# \(C::BlockCholesky{<:Real}, x::BlockVector) = ldiv!(C, copy(x))
# \(C::BlockCholesky{<:Real}, X::BlockMatrix) = ldiv!(C, copy(X))



# #
# # Adjoint and Transpose
# #

# function Base.adjoint(X::BlockMatrix)
#     return _BlockArray(
#         permutedims(collect(adjoint.(X.blocks))),
#         blocksizes(X, 2),
#         blocksizes(X, 1),
#     )
# end
# function Base.adjoint(x::BlockVector)
#     return _BlockArray(
#         permutedims(collect(adjoint.(x.blocks))),
#         [1],
#         blocksizes(x, 1),
#     )
# end

# function Base.transpose(X::BlockMatrix)
#     return _BlockArray(
#         permutedims(collect(transpose.(X.blocks))),
#         blocksizes(X, 2),
#         blocksizes(X, 1),
#     )
# end
# function Base.transpose(x::BlockVector)
#     return _BlockArray(
#         permutedims(collect(transpose.(x.blocks))),
#         [1],
#         blocksizes(x, 1),
#     )
# end



####################################### Multiplication #####################################


# function +(u::UniformScaling, X::AbstractBlockMatrix)
#     @assert cumulsizes(X, 1) == cumulsizes(X, 2)
#     Y = copy(X)
#     for p in 1:nblocks(Y, 1)
#         setblock!(Y, getblock(Y, p, p) + u, p, p)
#     end
#     return Y
# end
# +(u::UniformScaling, X::Symmetric{T, <:ABM{T}} where T) = Symmetric(u + unbox(X))
# function +(X::AbstractBlockMatrix, u::UniformScaling)
#     @assert cumulsizes(X, 1) == cumulsizes(X, 2)
#     Y = copy(X)
#     for p in 1:nblocks(Y, 1)
#         setblock!(Y, u + getblock(Y, p, p), p, p)
#     end
#     return Y
# end
# +(X::Symmetric{T, <:ABM{T}} where T, u::UniformScaling) = Symmetric(unbox(X) + u)

# # Define addition and subtraction for compatible block matrices and vectors.
# import Base: +, -
# for foo in [:+, :-]
#     @eval function $foo(A::BV{T}, B::BV{T}) where T
#         @assert blocksizes(A) == blocksizes(B)
#         C = similar(A)
#         for p in 1:nblocks(C, 1)
#             setblock!(C, $foo(getblock(A, p), getblock(B, p)), p)
#         end
#         return C
#     end
#     @eval function $foo(A::BM{T}, B::BM{T}) where T
#         @assert blocksizes(A) == blocksizes(B)
#         C = similar(A)
#         for q in 1:nblocks(C, 2), p in 1:nblocks(C, 1)
#             setblock!(C, $foo(getblock(A, p, q), getblock(B, p, q)), p, q)
#         end
#         return C
#     end
# end



# #################################### Broadcasting ##########################################
# # Override the usual broadcasting machinery for BlockArrays. This is a pretty disgusting
# # hack. At the time of writing it, I was also writing my first year report, so was short of
# # time. This is generally an open problem for AbstractBlockArrays which really does need to
# # be resolved at some point.

# # Very specific `broadcast` method for particular use case. Needs to be generalised.
# function broadcasted(f, A::AbstractBlockVector, b::Real)
#     return BlockVector([broadcast(f, getblock(A, p), b) for p in 1:nblocks(A, 1)])
# end
# function broadcasted(f, A::AbstractBlockMatrix, b::Real)
#     return BlockMatrix([broadcast(f, getblock(A, p, q), b)
#         for p in 1:nblocks(A, 1), q in 1:nblocks(A, 2)])
# end
