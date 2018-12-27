import Base: size, eachindex, getindex, view, ==, eltype, convert, zero
import Distances: pairwise
export ColsAreObs, BlockData



################################## Basic matrix data type ##################################

"""
    ColsAreObs{T, TX<:AbstractMatrix}

A lightweight box for an `AbstractMatrix` to make it behave like a vector of vectors.
"""
struct ColsAreObs{T, TX<:AbstractMatrix{T}} <: AbstractVector{Vector{T}}
    X::TX
    ColsAreObs(X::TX) where {T, TX<:AbstractMatrix{T}} = new{T, TX}(X)
end

@adjoint ColsAreObs(X::AbstractMatrix) = ColsAreObs(X), Δ->(Δ.X,)

==(D1::ColsAreObs, D2::ColsAreObs) = D1.X == D2.X
size(D::ColsAreObs) = (size(D.X, 2),)
getindex(D::ColsAreObs, n::Int) = D.X[:, n]
getindex(D::ColsAreObs, n::CartesianIndex{1}) = getindex(D, n[1])
getindex(D::ColsAreObs, n) = ColsAreObs(D.X[:, n])
view(D::ColsAreObs, n::Int) = view(D.X, :, n)
view(D::ColsAreObs, n) = ColsAreObs(view(D.X, :, n))
eltype(D::ColsAreObs{T}) where T = Vector{T}
zero(D::ColsAreObs) = ColsAreObs(zero(D.X))

const AdjColsAreObs{T, TX} = ColsAreObs{T, TX}

################################ Fancy block data set type #################################

"""
    BlockData{T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}} <: AbstractVector{T}

A strictly ordered collection of `AbstractVector`s, representing a ragged array of data.
"""
struct BlockData{T, V<:AbstractVector{<:T}} <: AbstractVector{T}
    X::Vector{V}
end
BlockData(X::Vector{V}) where {T, V<:AbstractVector{T}} = BlockData{T, V}(X)
BlockData(X::Vector{AbstractVector}) = BlockData{Any, AbstractVector}(X)
==(D1::BlockData, D2::BlockData) = D1.X == D2.X
size(D::BlockData) = (sum(length, D.X),)

blocks(X::BlockData) = X.X
function getindex(D::BlockData, n::Int)
    b = 1
    while n > length(D.X[b])
        n -= length(D.X[b])
        b += 1
    end
    return D.X[b][n]
end
function getindex(D::BlockData, n::BlockVector{<:Integer})
    @assert eachindex(D) == n
    return D
end
view(D::BlockData, b::Int, n) = view(D.X[b], n)
eltype(D::BlockData{T}) where T = T
function eachindex(D::BlockData)
    lengths = map(length, blocks(D))
    # return 1:sum(length, blocks(D))
    return BlockArray(1:sum(lengths), lengths)
    # return BlockData(lengths)
end

convert(::Type{BlockData}, x::BlockData) = x
convert(::Type{BlockData}, x::AbstractVector) = BlockData([x])
