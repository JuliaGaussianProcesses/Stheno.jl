import Base: size, eachindex, getindex, view, ==, endof, eltype, start, next, done,
    IndexStyle, map, convert, promote
import Distances: pairwise
export AbstractDataSet, ADS, BlockData

"""
    AbstractDataSet

Representation of a data set `D`, comprising `length(D)` observations.
"""
abstract type AbstractDataSet{T} <: AbstractVector{T} end

const ADS{T} = AbstractDataSet{T}

IndexStyle(::Type{<:AbstractDataSet}) = IndexLinear()



################################## Basic matrix data type ##################################

"""
    MatDataSet{T, TX<:AbstractMatrix}

A lightweight box for an `AbstractMatrix` to make it behave like a vector of vectors.
"""
struct MatDataSet{T, TX<:AbstractMatrix{T}} <: AbstractDataSet{Vector{T}}
    X::TX
    MatDataSet(X::TX) where {T, TX<:AbstractMatrix{T}} = new{T, TX}(X)
end

@inline Base.==(D1::MatDataSet, D2::MatDataSet) = D1.X == D2.X
@inline Base.size(D::MatDataSet) = (size(D.X, 2),)
@inline Base.getindex(D::MatDataSet, n::Int) = D.X[:, n]
@inline Base.getindex(D::MatDataSet, n) = MatDataSet(D.X[:, n])
@inline Base.Base.view(D::MatDataSet, n::Int) = view(D.X, :, n)
@inline Base.view(D::MatDataSet, n) = MatDataSet(view(D.X, :, n))
@inline Base.eltype(D::MatDataSet{T}) where T = Vector{T}



################################ Fancy block data set type #################################

"""
    BlockData{T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}} <: AbstractDataSet{T}

A strictly ordered collection of `AbstractVector`s, representing a ragged array of data.
"""
struct BlockData{T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}} <: AbstractDataSet{T}
    X::TX
    function BlockData(X::TX) where {T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}}
        return new{T, TV, TX}(X)
    end
end
@inline Base.==(D1::BlockData, D2::BlockData) = D1.X == D2.X
@inline Base.size(D::BlockData) = (sum(length, D.X),)

# HOW TO DO THIS CORRECTLY?

@inline blocks(X::BlockData) = X.X
@inline function Base.getindex(D::BlockData, n::Int)
    b = 1
    while n > length(D.X[b])
        n -= length(D.X[b])
        b += 1
    end
    return D.X[b][n]
end
@inline Base.view(D::BlockData, b::Int, n) = view(D.X[b], n)
@inline Base.eltype(D::BlockData{T}) where T = T



# ##################################### Generic conversion methods ###########################

# ADS(X::AVM) = DataSet(X)
# ADS(X::AV{<:ADS}) = BlockData(X)
# ADS(X::AV{<:AVM}) = BlockData(ADS.(X))
