const BlockDiagonal{T, TM} = BlockMatrix{T, <:Diagonal{TM}} where {TM <: AbstractMatrix{T}}



#
# Constructors
#

block_diagonal(vs::AbstractVector{<:AbstractMatrix}) = mortar(Diagonal(vs))

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

LinearAlgebra.adjoint(A::BlockDiagonal) = block_diagonal(adjoint.(A.blocks.diag))
LinearAlgebra.transpose(A::BlockDiagonal) = block_diagonal(transpose.(A.blocks.diag))


#
# UpperTriangular - ensure we get a BlockDiagonal back
#

function LinearAlgebra.UpperTriangular(A::BlockDiagonal)
    return block_diagonal(UpperTriangular.(A.blocks.diag))
end

function LinearAlgebra.LowerTriangular(A::BlockDiagonal)
    return block_diagonal(LowerTriangular.(A.blocks.diag))
end



#
# Symmetric - ensure we get a BlockDiagonal back
#

LinearAlgebra.Symmetric(A::BlockDiagonal) = block_diagonal(Symmetric.(A.blocks.diag))

ZygoteRules.@adjoint function LinearAlgebra.Symmetric(A::BlockDiagonal)
    return Zygote.pullback(A->block_diagonal(Symmetric.(A.blocks.diag)), A)
end



#
# Addition
#

function Base.:+(A::BlockDiagonal, B::BlockDiagonal)
    return block_diagonal([a + b for (a, b) in zip(A.blocks.diag, B.blocks.diag)])
end

function Base.:+(A::Matrix, B::BlockDiagonal)
    @assert size(A) == size(B)
    C = copy(A)
    cs = cumulsizes(B, 1)
    for n in 1:nblocks(B, 1)
        idx = cs[n]:cs[n+1]-1
        C[idx, idx] += B[Block(n, n)]
    end
    return C
end

ZygoteRules.@adjoint function Base.:+(A::Matrix, B::BlockDiagonal{T, <:Matrix{T}} where {T})
    return A + B, function(Δ)
        cs = cumulsizes(B, 1)
        blks = [Δ[cs[n]:cs[n+1]-1, cs[n]:cs[n+1]-1] for n in 1:nblocks(B, 1)]
        return (Δ, block_diagonal(blks))
    end
end


#
# Negation
#

Base.:-(A::BlockDiagonal) = block_diagonal([-a for a in A.blocks.diag])


#
# tr_At_A
#

tr_At_A(A::BlockDiagonal) = sum(tr_At_A.(A.blocks.diag))

ZygoteRules.@adjoint function tr_At_A(A::BlockDiagonal)
    return tr_At_A(A), Δ::Real->(block_diagonal(2Δ .* A.blocks.diag),)
end


#
# BlockDiagonal multiplication
#

function Base.:*(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    return block_diagonal([a * b for (a, b) in zip(A.blocks.diag, B.blocks.diag)])
end

ZygoteRules.@adjoint function Base.:*(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    return A * B, Δ->(Δ * B', A' * Δ)
end


function Base.:*(A::BlockDiagonal{<:Real}, B::Matrix{<:Real})
    A_blks, B_blks = A.blocks.diag, BlockArray(B, blocksizes(A, 1), [size(B, 2)]).blocks
    return Matrix(mortar(reshape([a * b for (a, b) in zip(A_blks, B_blks)], :, 1)))
end
function Base.:*(A::BlockDiagonal{<:Real}, x::Vector{<:Real})
    A_blks, x_blks = diag(A.blocks), BlockArray(x, blocksizes(A, 1)).blocks
    return Vector(mortar([a * x for (a, x) in zip(A_blks, x_blks)]))
end


#
# BlockDiagonal ldiv
#

function Base.:\(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    A_blks, B_blks = diag(A.blocks), diag(B.blocks)
    return block_diagonal([a \ b for (a, b) in zip(A_blks, B_blks)])
end

ZygoteRules.@adjoint function Base.:\(A::BlockDiagonal{<:Real}, B::BlockDiagonal{<:Real})
    Y = A \ B
    return Y, function(Ȳ::BlockDiagonal)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

function Base.:\(A::BlockDiagonal{<:Real}, B::AbstractMatrix{<:Real})
    A_blks = diag(A.blocks)
    B_blks = BlockArray(collect(B), blocksizes(A, 1), [size(B, 2)]).blocks

    return Matrix(mortar(reshape([a \ b for (a, b) in zip(A_blks, B_blks)], :, 1)))
end
ZygoteRules.@adjoint function Base.:\(A::BlockDiagonal{<:Real}, B::AbstractMatrix{<:Real})
    Y = A \ B
    return Y, function(Ȳ::AbstractMatrix{<:Real})
        B̄ = A' \ Ȳ
        return (_block_diag_bit(-B̄, Y', A), B̄)
    end
end

function Base.:\(A::BlockDiagonal{<:Real}, x::AbstractVector{<:Real})
    return reshape(A \ reshape(x, :, 1), :)
end

ZygoteRules.@adjoint function Base.:\(A::BlockDiagonal{<:Real}, x::AbstractVector{<:Real})
    y_mat, back = Zygote.pullback(\, A, reshape(x, :, 1))
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

function LinearAlgebra.cholesky(A::BlockDiagonal{<:Real})
    Cs = map(A->cholesky(A).U, diag(A.blocks))
    return Cholesky(block_diagonal(Cs), :U, 0)
end
ZygoteRules.@adjoint function LinearAlgebra.cholesky(A::BlockDiagonal{<:Real})
    Cs_backs = map(A->Zygote.pullback(A->cholesky(A).U, A), diag(A.blocks))
    Cs, backs = first.(Cs_backs), last.(Cs_backs)
    function back(Ū::BlockDiagonal)
        # @show typeof(map((Ū, back)->first(back(Ū)), diag(Ū.blocks), backs))
        return (block_diagonal(map((Ū, back)->first(back(Ū)), diag(Ū.blocks), backs)),)
    end
    return Cholesky(block_diagonal(Cs), :U, 0), Δ->back(Δ.factors)
end

function LinearAlgebra.logdet(C::Cholesky{T, <:BlockDiagonal{T}} where {T<:Real})
    return 2 * sum([logdet(c) for c in C.factors.blocks.diag])
end

ZygoteRules.@adjoint function LinearAlgebra.logdet(
    C::Cholesky{T, <:BlockDiagonal{T}} where {T<:Real},
)
    return logdet(C), function(Δ::Real)
        blks = C.factors.blocks.diag
        factors = block_diagonal([diagm(0=>2Δ ./ diag(b)) for b in blks])
        return ((factors=factors,),)
    end
end


#
# Misc
#

# Legacy version of logabsdet for pre-1.4.
if VERSION < v"1.4.0"
    LinearAlgebra.logabsdet(d::Diagonal) = logabsdet(UpperTriangular(d))
end

#
# BlockDiagonal mul! and *
#

Base.:*(D::BlockDiagonal, x::BlockVector) = mul!(copy(x), D, x)
Base.:*(D::BlockDiagonal, X::BlockMatrix) = mul!(copy(X), D, X)

function LinearAlgebra.mul!(y::BlockVector, D::BlockDiagonal, x::BlockVector)
    @assert are_conformal(D, x) && are_conformal(D, y)
    blocks = D.blocks.diag
    for r in 1:nblocks(D, 1)
        mul!(getblock(y, r), blocks[r], getblock(x, r))
    end
    return y
end

@adjoint function Base.:*(D::BlockDiagonal, x::BlockVector)
    y = D * x
    return y, function(ȳ::BlockVector)
        @assert blocksizes(y, 1) == blocksizes(ȳ, 1)
        D̄_blocks = map((x_blk, ȳ_blk) -> x_blk * ȳ_blk', x.blocks, ȳ.blocks)
        D̄ = mortar(Diagonal(D̄_blocks))
        return D̄, D' * ȳ
    end
end

function LinearAlgebra.mul!(Y::BlockMatrix, D::BlockDiagonal, X::BlockMatrix)
    @assert are_conformal(D, X) && are_conformal(D, Y)
    blocks = D.blocks.diag
    for r in 1:nblocks(D, 1)
        for c in 1:nblocks(X, 2)
            mul!(getblock(Y, r, c), blocks[r], getblock(X, r, c))
        end
    end
    return Y
end
