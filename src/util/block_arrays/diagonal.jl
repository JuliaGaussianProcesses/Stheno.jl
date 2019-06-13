const BlockDiagonal{T, TM} = BlockMatrix{T, <:Diagonal{TM}} where {TM <: AbstractMatrix{T}}
const UpperTriangularBlockDiagonal{T} = UpperTriangular{T, <:BlockDiagonal{T}} where {T}

import Base: +



#
# Constructors
#

function block_diagonal(vs::AbstractVector{<:AbstractMatrix})
    return _BlockArray(Diagonal(vs), size.(vs, 1), size.(vs, 2))
end

function LinearAlgebra.diagzero(D::Diagonal{<:AbstractMatrix{T}}, r, c) where {T}
    return Zeros{T}(size(D.diag[r], 1), size(D.diag[c], 2))
end



#
# Addition
#

function +(A::Matrix, B::BlockDiagonal)
    @assert size(A) == size(B)
    C = copy(A)
    cs = cumulsizes(B, 1)
    for n in 1:nblocks(B, 1)
        idx = cs[n]:cs[n+1]-1
        C[idx, idx] += B[Block(n, n)]
    end
    return C
end
@adjoint function +(A::Matrix, B::BlockDiagonal{T, <:Matrix{T}} where {T})
    return A + B, function(Δ)
        cs = cumulsizes(B, 1)
        blks = [Δ[cs[n]:cs[n+1]-1, cs[n]:cs[n+1]-1] for n in 1:nblocks(B, 1)]
        return (Δ, block_diagonal(blks))
    end
end



#
# BlockDiagonal mul! and *
#

*(D::BlockDiagonal, x::BlockVector) = mul!(copy(x), D, x)
*(D::BlockDiagonal, X::BlockMatrix) = mul!(copy(X), D, X)

function mul!(y::BlockVector, D::BlockDiagonal, x::BlockVector)
    @assert are_conformal(D, x) && are_conformal(D, y)
    blocks = D.blocks.diag
    for r in 1:nblocks(D, 1)
        mul!(getblock(y, r), blocks[r], getblock(x, r))
    end
    return y
end

@adjoint function *(D::BlockDiagonal, x::BlockVector)
    y = D * x
    return y, function(ȳ::BlockVector)
        @assert blocksizes(y, 1) == blocksizes(ȳ, 1)
        D̄_blocks = map((x_blk, ȳ_blk) -> x_blk * ȳ_blk', x.blocks, ȳ.blocks)
        D̄ = _BlockArray(Diagonal(D̄_blocks), blocksizes(D, 1), blocksizes(D, 2))
        return D̄, D' * ȳ
    end
end

function mul!(Y::BlockMatrix, D::BlockDiagonal, X::BlockMatrix)
    @assert are_conformal(D, X) && are_conformal(D, Y)
    blocks = D.blocks.diag
    for r in 1:nblocks(D, 1)
        for c in 1:nblocks(X, 2)
            mul!(getblock(Y, r, c), blocks[r], getblock(X, r, c))
        end
    end
    return Y
end

@adjoint function *(D::BlockDiagonal, X::BlockMatrix)
    error("Not implemented")
    Y = D * X
    return Y, function(Ȳ)
        @assert blocksizes(Y) == blocksizes(Ȳ)
        blocks = D.blocks.diag

    end
end



#
# UpperTriangularBlockDiagonal ldiv! and \
#

function ldiv!(U::UpperTriangularBlockDiagonal, x::BlockVector)
    @assert are_conformal(U.data, x)
    blocks = U.data.blocks.diag
    for n in 1:nblocks(x, 1)
        setblock!(x, ldiv!(blocks[n], x[Block(n)]), n)
    end
    return x
end
\(U::UpperTriangularBlockDiagonal, x::BlockVector) = ldiv!(U, copy(x))

function ldiv!(U::UpperTriangularBlockDiagonal, X::BlockMatrix)
    @assert are_conformal(U.data, X)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(X, 1)
        for c in 1:nblocks(X, 2)
            setblock!(X, ldiv!(blocks[r], X[Block(r, c)]), r, c)
        end
    end
    return X
end
\(U::UpperTriangularBlockDiagonal, X::BlockMatrix) = ldiv!(U, copy(X))



#
# UpperTriangularBlockDiagonal mul! and *
#

*(D::UpperTriangularBlockDiagonal, x::BlockVector) = mul!(copy(x), D, x)
*(D::UpperTriangularBlockDiagonal, X::BlockMatrix) = mul!(copy(X), D, X)

function mul!(y::BlockVector, U::UpperTriangularBlockDiagonal, x::BlockVector)
    @assert are_conformal(U.data, x) && are_conformal(U.data, y)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(U.data, 1)
        mul!(getblock(y, r), UpperTriangular(blocks[r]), getblock(x, r))
    end
    return y
end

function mul!(Y::BlockMatrix, U::UpperTriangularBlockDiagonal, X::BlockMatrix)
    @assert are_conformal(U.data, X) && are_conformal(U.data, Y)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(U.data, 1)
        for c in 1:nblocks(X, 2)
            mul!(getblock(Y, r, c), UpperTriangular(blocks[r]), getblock(X, r, c))
        end
    end
    return Y
end



#
# cholesky
#

# Strip unhelpful wrappers to ensure that ldiv! is efficient.
strip_block(X::UpperTriangular) = X
strip_block(X::UpperTriangular{T, <:Diagonal{T}} where {T}) = X.data

function cholesky(A::BlockDiagonal{<:Real})
    Cs = [strip_block(cholesky(A).U) for A in diag(A.blocks)]
    return Cholesky(_BlockArray(Diagonal(Cs), A.block_sizes), :U, 0)
end
function cholesky(A::Symmetric{T, <:BlockDiagonal{T}} where {T<:Real})
    Cs = [strip_block(cholesky(Symmetric(A)).U) for A in diag(A.data.blocks)]
    return Cholesky(_BlockArray(Diagonal(Cs), A.data.block_sizes), :U, 0)
end

function logdet(C::Cholesky{T, <:BlockDiagonal{T}} where {T<:Real})
    return 2 * sum(n->logabsdet(C.factors[Block(n, n)])[1], 1:nblocks(C.factors, 1))
end



#
# Misc
#

# Because Base is dumb and hasn't implemented `logabsdet` for `Diagonal` matrices.
logabsdet(d::Diagonal) = logabsdet(UpperTriangular(d))
