import Base: size, eachindex, getindex, view, ==, eltype

"""
    BlockData{T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}} <: AbstractVector{T}

A strictly ordered collection of `AbstractVector`s, representing a ragged array of data.
"""
struct BlockData{T, V<:AbstractVector{<:T}} <: AbstractVector{T}
    X::Vector{V}
end

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

view(D::BlockData, b::Int, n) = view(D.X[b], n)

eltype(D::BlockData{T}) where {T} = T

function eachindex(D::BlockData)
    lengths = map(length, blocks(D))
    return BlockArray(1:sum(lengths), lengths)
end
