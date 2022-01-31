"""
    GPPPInput(p, x::AbstractVector)

An collection of inputs for a GPPP.
`p` indicates which process the vector `x` should be extracted from.
The required type of `p` is determined by the type of the keys in the `GPPP` indexed.

```jldoctest
julia> x = [1.0, 1.5, 0.3];

julia> v = GPPPInput(:a, x)
3-element GPPPInput{Symbol, Float64, Vector{Float64}}:
 (:a, 1.0)
 (:a, 1.5)
 (:a, 0.3)

julia> v isa AbstractVector{Tuple{Symbol, Float64}}
true

julia> v == map(x_ -> (:a, x_), x)
true
```
"""
struct GPPPInput{Tp, T, Tx<:AbstractVector{T}} <: AbstractVector{Tuple{Tp, T}}
    p::Tp
    x::Tx
end

Base.size(x::GPPPInput) = (length(x.x), )

Base.getindex(x::GPPPInput, idx::Integer) = only(x[idx:idx])

Base.getindex(x::GPPPInput, idx) = map(x_ -> (x.p, x_), x.x[idx])



"""
    BlockData{T, TV<:AbstractVector{T}, TX<:AbstractVector{TV}} <: AbstractVector{T}

A strictly ordered collection of `AbstractVector`s, representing a ragged array of data.

Very useful when working with `GPPP`s. For example
```julia
f = @gppp let
    f1 = GP(SEKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end

# Specify a `BlockData` set that can be used to index into
# the `f2` and `f3` processes in `f`.
x = BlockData(
    GPPPInput(:f2, randn(4)),
    GPPPInput(:f3, randn(3)),
)

# Index into `f` at the input.
f(x)
```
"""
struct BlockData{T, V<:AbstractVector{<:T}} <: AbstractVector{T}
    X::Vector{V}
end

BlockData(X::Vector{AbstractVector}) = BlockData{Any, AbstractVector}(X)

BlockData(xs::AbstractVector...) = BlockData([xs...])

Base.size(D::BlockData) = (sum(length, D.X),)

function Base.getindex(D::BlockData, n::Int)
    b = 1
    while n > length(D.X[b])
        n -= length(D.X[b])
        b += 1
    end
    return D.X[b][n]
end

Base.:(==)(D1::BlockData, D2::BlockData) = D1.X == D2.X

blocks(X::BlockData) = X.X

Base.view(D::BlockData, b::Int, n) = view(D.X[b], n)

Base.eltype(D::BlockData{T}) where {T} = T

function Base.eachindex(D::BlockData)
    lengths = map(length, blocks(D))
    return BlockArray(1:sum(lengths), lengths)
end

Base.vcat(x::GPPPInput...) = BlockData(AbstractVector[x...])

Base.vcat(x::GPPPInput{Symbol, T}...) where {T} = BlockData([x...])
