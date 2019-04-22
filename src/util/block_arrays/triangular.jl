import LinearAlgebra: mul!, *, ldiv!, \, adjoint, transpose

const BlockLowerTriangular{T} = LowerTriangular{T, <:BlockMatrix{T}}
const BlockUpperTriangular{T} = UpperTriangular{T, <:BlockMatrix{T}}

#
# adjoint / transpose
#

adjoint(X::BlockLowerTriangular) = UpperTriangular(adjoint(X.data))
transpose(X::BlockLowerTriangular) = UpperTriangular(transpose(X.data))

adjoint(X::BlockUpperTriangular) = LowerTriangular(adjoint(X.data))
transpose(X::BlockUpperTriangular) = LowerTriangular(transpose(X.data))

#
# mul! and *
#

function mul!(y::BlockVector, L::BlockLowerTriangular, x::BlockVector)
    error("Not implemented")
end

function mul!(Y::BlockMatrix, L::BlockLowerTriangular, X::BlockMatrix)
    error("Not implemented")
end

function mul!(y::BlockVector, U::BlockUpperTriangular, x::BlockVector)
    error("Not implemented")
end

function mul!(Y::BlockMatrix, U::BlockUpperTriangular, X::BlockMatrix)
    error("Not implemented")
end 

*(L::BlockLowerTriangular, x::BlockVector) = error("Not implemented")
*(L::BlockLowerTriangular, X::BlockMatrix) = error("Not implemented")
*(U::BlockUpperTriangular, x::BlockVector) = error("Not implemented")
*(U::BlockUpperTriangular, X::BlockMatrix) = error("Not implemented")

#
# ldiv! and \
#

function ldiv!(L::BlockLowerTriangular, x::BlockVector)
    error("Not implemented")
end

function ldiv!(L::BlockLowerTriangular, X::BlockMatrix)
    error("Not implemented")
end

function ldiv!(U::BlockUpperTriangular, x::BlockVector)
    error("Not implemented")
end

function ldiv!(U::BlockUpperTriangular, X::BlockMatrix)
    error("Not implemented")
end

\(L::BlockLowerTriangular, x::BlockVector) = ldiv!(L, copy(x))
\(L::BlockLowerTriangular, X::BlockMatrix) = ldiv!(L, copy(X))
\(U::BlockUpperTriangular, x::BlockVector) = ldiv!(U, copy(x))
\(U::BlockUpperTriangular, X::BlockMatrix) = ldiv!(U, copy(X))


# function _block_ldiv_mat_upper(U, X)
#     @assert are_conformal(unbox(U), X)
#     Y = BlockMatrix{eltype(U)}(undef_blocks, blocksizes(U, 1), blocksizes(X, 2))
#     for q in 1:nblocks(Y, 2), p in reverse(1:nblocks(Y, 1))
#         setblock!(Y, getblock(X, p, q), p, q)
#         for p′ in p+1:nblocks(Y, 1)
#             setblock!(Y, getblock(Y, p, q) - getblock(U, p, p′) * getblock(Y, p′, q), p, q)
#         end
#         setblock!(Y, getblock(U, p, p) \ getblock(Y, p, q), p, q)
#     end
#     return Y
# end
