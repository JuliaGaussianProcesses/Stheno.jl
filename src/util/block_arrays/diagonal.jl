const BlockDiagonal{T, TM} = BlockMatrix{T, <:Diagonal{TM}} where {TM <: AbstractMatrix{T}}

import Base: +, -, adjoint, transpose
import LinearAlgebra: UpperTriangular, LowerTriangular, Symmetric


#
# Constructors
#

function block_diagonal(vs::AbstractVector{<:AbstractMatrix})
    return _BlockArray(Diagonal(vs), size.(vs, 1), size.(vs, 2))
end

function LinearAlgebra.diagzero(D::Diagonal{<:AbstractMatrix{T}}, r, c) where {T}
    return zeros(T, size(D.diag[r], 1), size(D.diag[c], 2))
end


#
# Accumulation rule for Zygote.
#

function Zygote.accum(A::BlockDiagonal, B::BlockDiagonal)
    return block_diagonal(accum.(A.blocks.diag, B.blocks.diag))
end


#
# adjoint / transpose - ensure we get a BlockDiagonal back
#

adjoint(A::BlockDiagonal) = block_diagonal(adjoint.(A.blocks.diag))
transpose(A::BlockDiagonal) = block_diagonal(transpose.(A.blocks.diag))


#
# UpperTriangular - ensure we get a BlockDiagonal back
#

UpperTriangular(A::BlockDiagonal) = block_diagonal(UpperTriangular.(A.blocks.diag))
LowerTriangular(A::BlockDiagonal) = block_diagonal(LowerTriangular.(A.blocks.diag))


#
# Symmetric - ensure we get a BlockDiagonal back
#

Symmetric(A::BlockDiagonal) = block_diagonal(Symmetric.(A.blocks.diag))
@adjoint function Symmetric(A::BlockDiagonal)
    return Zygote.forward(A->block_diagonal(Symmetric.(A.blocks.diag)), A)
end


#
# Addition
#

function +(A::BlockDiagonal, B::BlockDiagonal)
    return block_diagonal([a + b for (a, b) in zip(A.blocks.diag, B.blocks.diag)])
end

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
# Negation
#

-(A::BlockDiagonal) = block_diagonal([-a for a in A.blocks.diag])


#
# tr_At_A
#

tr_At_A(A::BlockDiagonal) = sum(tr_At_A.(A.blocks.diag))

@adjoint function tr_At_A(A::BlockDiagonal)
    return tr_At_A(A), Δ::Real->(block_diagonal(2Δ .* A.blocks.diag),)
end


#
# BlockDiagonal multiplication
#

function *(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    return block_diagonal([a * b for (a, b) in zip(A.blocks.diag, B.blocks.diag)])
end

@adjoint *(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real}) = A * B, Δ->(Δ * B', A' * Δ)


function *(A::BlockDiagonal{<:Real}, B::Matrix{<:Real})
    A_blks, B_blks = A.blocks.diag, BlockArray(B, blocksizes(A, 1), [size(B, 2)]).blocks
    return Matrix(BlockMatrix(reshape([a * b for (a, b) in zip(A_blks, B_blks)], :, 1)))
end
function *(A::BlockDiagonal{<:Real}, x::Vector{<:Real})
    A_blks, x_blks = diag(A.blocks), BlockArray(x, blocksizes(A, 1)).blocks
    return Vector(BlockVector([a * x for (a, x) in zip(A_blks, x_blks)]))
end


#
# BlockDiagonal ldiv
#

function \(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    A_blks, B_blks = diag(A.blocks), diag(B.blocks)
    return block_diagonal([a \ b for (a, b) in zip(A_blks, B_blks)])
end

@adjoint function \(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    Y = A \ B
    return Y, function(Ȳ::BlockDiagonal)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

function \(A::BlockDiagonal{<:Real}, B::AbstractMatrix{<:Real})
    A_blks, B_blks = diag(A.blocks), BlockArray(collect(B), blocksizes(A, 1), [size(B, 2)]).blocks
    return Matrix(BlockMatrix(reshape([a \ b for (a, b) in zip(A_blks, B_blks)], :, 1)))
end
@adjoint function \(A::BlockDiagonal{<:Real}, B::AbstractMatrix{<:Real})
    Y = A \ B
    return Y, function(Ȳ::AbstractMatrix{<:Real})
        B̄ = A' \ Ȳ
        return (_block_diag_bit(-B̄, Y', A), B̄)
    end
end

\(A::BlockDiagonal{<:Real}, x::AbstractVector{<:Real}) = reshape(A \ reshape(x, :, 1), :)
@adjoint function \(A::BlockDiagonal{<:Real}, x::AbstractVector{<:Real})
    y_mat, back = Zygote.forward(\, A, reshape(x, :, 1))
    return vec(y_mat), function(Δ::AbstractVector{<:Real})
        Ā, x̄ = back(reshape(Δ, :, 1))
        return Ā, vec(x̄)
    end
end

function _block_diag_bit(A::AbstractMatrix, B::AbstractMatrix, R::BlockDiagonal)
    A_blks = vec(BlockArray(A, blocksizes(R, 1), [size(A, 2)]).blocks)
    B_blks = vec(BlockArray(B', blocksizes(R, 1), [size(B, 1)]).blocks)
    return block_diagonal([a * b' for (a, b) in zip(A_blks, B_blks)])
end


#
# cholesky
#

function cholesky(A::BlockDiagonal{<:Real})
    Cs = map(A->cholesky(A).U, diag(A.blocks))
    return Cholesky(block_diagonal(Cs), :U, 0)
end
@adjoint function cholesky(A::BlockDiagonal{<:Real})
    Cs_backs = map(A->Zygote.forward(A->cholesky(A).U, A), diag(A.blocks))
    Cs, backs = first.(Cs_backs), last.(Cs_backs)
    function back(Ū::BlockDiagonal)
        # @show typeof(map((Ū, back)->first(back(Ū)), diag(Ū.blocks), backs))
        return (block_diagonal(map((Ū, back)->first(back(Ū)), diag(Ū.blocks), backs)),)
    end
    return Cholesky(block_diagonal(Cs), :U, 0), Δ->back(Δ.factors)
end

function logdet(C::Cholesky{T, <:BlockDiagonal{T}} where {T<:Real})
    return 2 * sum([logdet(c) for c in C.factors.blocks.diag])
end

@adjoint function logdet(C::Cholesky{T, <:BlockDiagonal{T}} where {T<:Real})
    return logdet(C), function(Δ::Real)
        blks = C.factors.blocks.diag
        factors = block_diagonal([diagm(0=>2Δ ./ diag(b)) for b in blks])
        return ((factors=factors,),)
    end
end


#
# Misc
#

# Because Base is dumb and hasn't implemented `logabsdet` for `Diagonal` matrices.
logabsdet(d::Diagonal) = logabsdet(UpperTriangular(d))


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
