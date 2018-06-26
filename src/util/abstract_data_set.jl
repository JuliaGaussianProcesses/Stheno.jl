import Base: size, eachindex, getindex, view, ==, endof, eltype, start, next, done,
    IndexStyle, map, convert, promote
import Distances: pairwise
export AbstractDataSet, ADS, DataSet, BlockData

"""
    AbstractDataSet

Representation of a data set `D`, comprising `length(D)` observations.
"""
abstract type AbstractDataSet{T} <: AbstractVector{T} end

const ADS{T} = AbstractDataSet{T}

IndexStyle(::Type{<:AbstractDataSet}) = IndexLinear()



################################## Basic data type ##################################

"""
    DataSet{TX<:AbstractArray}

A light-weight wrapper around an `AbstractArray` to make it behave like a linearly-
indexed list of data. Is almost identical in behaviour to an `AbstractVector`. If wrapping
an `AbstractMatrix`, then the nth element of is the nth column of the matrix.
"""
struct DataSet{T, TX<:AbstractArray{T}} <: AbstractDataSet{T}
    X::TX
    function DataSet(X::TX) where {T, TX<:AbstractArray{T}}
        @assert ndims(X) < 3
        return new{T, TX}(X)
    end
end

const VectorData{T} = DataSet{T, <:AbstractVector{T}}
const MatrixData{T} = DataSet{T, <:AbstractMatrix{T}}

@inline ==(D1::DataSet, D2::DataSet) = D1.X == D2.X
@inline endof(D::DataSet) = length(D)
@inline start(::DataSet) = 1
@inline done(D::DataSet, n::Int) = n > length(D) ? true : false

# Vector data set.
@inline size(D::VectorData) = (length(D.X),)
@inline getindex(D::VectorData, n::Int) = D.X[n]
@inline getindex(D::VectorData, n) = DataSet(D.X[n])
@inline view(D::VectorData, n::Int) = view(D.X, n)
@inline view(D::VectorData, n) = DataSet(view(D.X, n))
@inline eltype(D::VectorData{T}) where T = T
@inline next(D::VectorData, n::Int) = (D.X[n], n + 1)

# Matrix data set.
@inline size(D::MatrixData) = (size(D.X, 2),)
@inline getindex(D::MatrixData, n::Int) = D.X[:, n]
@inline getindex(D::MatrixData, n) = DataSet(D.X[:, n])
@inline view(D::MatrixData, n::Int) = view(D.X, :, n)
@inline view(D::MatrixData, n) = DataSet(view(D.X, :, n))
@inline eltype(D::MatrixData{T}) where T = Vector{T}
@inline next(D::MatrixData, n::Int) = (view(D, n), n + 1)



################################ Fancy block data set type #################################

"""
    BlockData{TX<:AbstractVector{<:AbstractDataSet}} <: AbstractDataSet    

A strictly ordered collection of `AbstractDataSet`s.
"""
struct BlockData{TX<:AbstractVector{<:AbstractDataSet}} <: AbstractDataSet{Any}
    X::TX
end
function BlockData(D::AbstractVector{<:AbstractArray})
    @assert all(ndims.(D) .< 3)
    return BlockData(DataSet.(D))
end
@inline ==(D1::BlockData, D2::BlockData) = D1.X == D2.X
@inline size(D::BlockData) = (sum(length, D.X),)
@inline endof(D::BlockData) = (length(D.X), length(D.X[end]))

@inline blocks(X::BlockData) = X.X
@inline function getindex(D::BlockData, n::Int)
    b = 1
    while n > length(D.X[b])
        n -= length(D.X[b])
        b += 1
    end
    return D.X[b][n]
end
@inline getindex(D::BlockData, b::Int, n) = D.X[b][n]
@inline view(D::BlockData, b::Int, n) = view(D.X[b], n)
@inline eltype(D::BlockData) = Any

@inline start(D::BlockData) = (1, start(D.X[1]))
function next(D::BlockData, state::Tuple{Int, Int})
    b, n = state
    x = D.X[b]
    if !done(x, n)
        val, n′ = next(x, n)
        return val, (b, n′)
    else
        b′ = b + 1
        val, n′ = next(D.X[b′], start(D.X[b′]))
        return val, (b′, n′)
    end 
end
@inline function done(D::BlockData, state::Tuple{Int, Int})
    b, n = state
    return b == length(D.X) && done(D.X[b], n)
end



##################################### Generic conversion methods ###########################

ADS(X::AVM) = DataSet(X)
ADS(X::AV{<:ADS}) = BlockData(X)
ADS(X::AV{<:AVM}) = BlockData(ADS.(X))

