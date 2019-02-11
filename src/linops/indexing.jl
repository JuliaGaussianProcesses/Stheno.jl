"""
    (f::GP)(x::AbstractVector)
    (f::BlockGP)(x::BlockGP)

Construct a `FiniteGP` representing the projection of `f` at `x`.
"""
(f::GP)(x...) = index_gp(f, x...)
(f::BlockGP)(x...) = index_gp(f, x...)

index_gp(f::AbstractGP, x::AV, σ²::AV{<:Real}) = FiniteGP(f, x, σ²)
index_gp(f::AbstractGP, x::AV, σ²::Real) = index_gp(f, x, Fill(σ², length(x)))
index_gp(f::AbstractGP, x::AV) = index_gp(f, x, Zeros{Int}(length(x)))
