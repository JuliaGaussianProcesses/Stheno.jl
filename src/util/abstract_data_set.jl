import Base: size, eachindex, getindex, view, ==, eltype
import Distances: pairwise
export MatData, BlockData



################################## Basic matrix data type ##################################

"""
    MatData{T, TX<:AbstractMatrix}

A lightweight box for an `AbstractMatrix` to make it behave like a vector of vectors.
"""
struct MatData{T, TX<:AbstractMatrix{T}} <: AbstractVector{Vector{T}}
    X::TX
    MatData(X::TX) where {T, TX<:AbstractMatrix{T}} = new{T, TX}(X)
end

@inline ==(D1::MatData, D2::MatData) = D1.X == D2.X
@inline size(D::MatData) = (size(D.X, 2),)
@inline getindex(D::MatData, n::Int) = D.X[:, n]
@inline getindex(D::MatData, n) = MatData(D.X[:, n])
@inline view(D::MatData, n::Int) = view(D.X, :, n)
@inline view(D::MatData, n) = MatData(view(D.X, :, n))
@inline eltype(D::MatData{T}) where T = Vector{T}



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
@inline ==(D1::BlockData, D2::BlockData) = D1.X == D2.X
@inline size(D::BlockData) = (sum(length, D.X),)

@inline blocks(X::BlockData) = X.X
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
@inline view(D::BlockData, b::Int, n) = view(D.X[b], n)
@inline eltype(D::BlockData{T}) where T = T
function eachindex(D::BlockData)
    lengths = map(length, blocks(D))
    return BlockArray(1:sum(lengths), lengths)
end

