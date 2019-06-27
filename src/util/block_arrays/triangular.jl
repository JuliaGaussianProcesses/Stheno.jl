# import LinearAlgebra: mul!, *, ldiv!, \, adjoint, transpose
# import BlockArrays: BlockArray



# #
# # constructors
# #

# function BlockArray(L::LowerTriangular, Ps::Vararg{AbstractVector{Int}, 2})
#     return LowerTriangular(BlockArray(L.data, Ps...))
# end
# function BlockArray(U::UpperTriangular, Ps::Vararg{AbstractVector{Int}, 2})
#     return UpperTriangular(BlockArray(U.data, Ps...))
# end



# #
# # adjoint / transpose
# #

# adjoint(X::BlockLowerTriangular) = UpperTriangular(adjoint(X.data))
# transpose(X::BlockLowerTriangular) = UpperTriangular(transpose(X.data))

# adjoint(X::BlockUpperTriangular) = LowerTriangular(adjoint(X.data))
# transpose(X::BlockUpperTriangular) = LowerTriangular(transpose(X.data))



# #
# # mul! and *
# #

# function mul!(y::BlockVector, L::BlockLowerTriangular, x::BlockVector)
#     @assert are_conformal(L, x) && are_conformal(L', y)

#     for r in 1:nblocks(L, 1)
#         fill!(y[Block(r)], 0.0)
#         for c in 1:r-1
#             y[Block(r)] = y[Block(r)] + L[Block(r, c)] * x[Block(c)]
#         end
#         y[Block(r)] = y[Block(r)] + LowerTriangular(L[Block(r, r)]) * x[Block(r)]
#     end
#     return y
# end

# function mul!(Y::BlockMatrix, L::BlockLowerTriangular, X::BlockMatrix)
#     @assert are_conformal(L, X)
#     @assert cumulsizes(L, 1) == cumulsizes(Y, 1)
#     @assert cumulsizes(Y, 2) == cumulsizes(X, 2)

#     P, Q, R = nblocks(L, 1), nblocks(L, 2), nblocks(X, 2)
#     for p in 1:P, r in 1:R
#         fill!(Y[Block(p, r)], 0.0)
#         for q in 1:p-1
#             Y[Block(p, r)] = Y[Block(p, r)] + L[Block(p, q)] * X[Block(q, r)]
#         end
#         Y[Block(p, r)] = Y[Block(p, r)] + LowerTriangular(L[Block(p, p)]) * X[Block(p, r)]
#     end
#     return Y
# end

# function mul!(y::BlockVector, U::BlockUpperTriangular, x::BlockVector)
#     @assert are_conformal(U, x) && are_conformal(U', y)

#     for r in 1:nblocks(U, 1)
#         mul!(y[Block(r)], UpperTriangular(U[Block(r, r)]), x[Block(r)])
#         for c in r+1:nblocks(U, 2)
#             y[Block(r)] = y[Block(r)] + U[Block(r, c)] * x[Block(c)]
#         end
#     end
#     return y
# end

# function mul!(Y::BlockMatrix, U::BlockUpperTriangular, X::BlockMatrix)
#     @assert are_conformal(U, X)
#     @assert cumulsizes(U, 1) == cumulsizes(Y, 1)
#     @assert cumulsizes(Y, 2) == cumulsizes(X, 2)

#     P, Q, R = nblocks(U, 1), nblocks(U, 2), nblocks(X, 2)
#     for p in 1:P, r in 1:R
#         mul!(Y[Block(p, r)], UpperTriangular(U[Block(p, p)]), X[Block(p, r)])
#         for q in p+1:Q
#             Y[Block(p, r)] = Y[Block(p, r)] + U[Block(p, q)] * X[Block(q, r)]
#         end
#     end
#     return Y
# end 

# *(A::BlockTriangular, x::BlockVector) = mul!(copy(x), A, x)
# *(A::BlockTriangular, X::BlockMatrix) = mul!(copy(X), A, X)

# @adjoint function *(A::BlockTriangular, x::BlockVector)
#     back(Δ::BlockVector) = (Δ * x', A' * Δ)
#     back(Δ::AbstractVector) = back(BlockArray(Δ, blocksizes(x)))
#     return A * x, back
# end
# @adjoint function *(A::BlockTriangular, B::BlockMatrix)
#     back(Δ::BlockMatrix) = (Δ * B', A' * Δ)
#     back(Δ::AbstractMatrix) = back(BlockArray(Δ, blocksizes(B)))
#     return A * B, back
# end



# #
# # ldiv! and \
# #

# function ldiv!(L::BlockLowerTriangular, x::BlockVector)
#     @assert are_conformal(L, x)
#     for p in 1:nblocks(x, 1)
#         for p′ in 1:p-1
#             x[Block(p)] = x[Block(p)] - L[Block(p, p′)] * x[Block(p′)]
#         end
#         ldiv!(LowerTriangular(L[Block(p, p)]), x[Block(p)])
#     end
#     return x
# end

# function ldiv!(L::BlockLowerTriangular, X::BlockMatrix)
#     @assert are_conformal(L, X)
#     for q in 1:nblocks(X, 2), p in 1:nblocks(X, 1)
#         for p′ in 1:p-1
#             X[Block(p, q)] = X[Block(p, q)] - L[Block(p, p′)] * X[Block(p′, q)]
#         end
#         ldiv!(LowerTriangular(L[Block(p, p)]), X[Block(p, q)])
#     end
#     return X
# end

# function ldiv!(U::BlockUpperTriangular, x::BlockVector)
#     @assert are_conformal(U, x )
#     for p in reverse(1:nblocks(x, 1))
#         for p′ in p+1:nblocks(x, 1)
#             x[Block(p)] = x[Block(p)] - U[Block(p, p′)] * x[Block(p′)]
#         end
#         ldiv!(UpperTriangular(U[Block(p, p)]), x[Block(p)])
#     end
#     return x
# end

# function ldiv!(U::BlockUpperTriangular, X::BlockMatrix)
#     @assert are_conformal(U, X)
#     for q in 1:nblocks(X, 2), p in reverse(1:nblocks(X, 1))
#         for p′ in p+1:nblocks(X, 1)
#             X[Block(p, q)] = X[Block(p, q)] - U[Block(p, p′)] * X[Block(p′, q)]
#         end
#         ldiv!(UpperTriangular(U[Block(p, p)]), X[Block(p, q)])
#     end
#     return X
# end

# \(L::BlockLowerTriangular, x::BlockVector) = ldiv!(L, copy(x))
# \(L::BlockLowerTriangular, X::BlockMatrix) = ldiv!(L, copy(X))
# \(U::BlockUpperTriangular, x::BlockVector) = ldiv!(U, copy(x))
# \(U::BlockUpperTriangular, X::BlockMatrix) = ldiv!(U, copy(X))
