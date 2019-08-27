"""
    (f::AbstractGP)(x::AbstractVector)

Construct a `FiniteGP` representing the projection of `f` at `x`.
"""
(f::GP)(x...) = FiniteGP(f, x...)
(f::CompositeGP)(x...) = FiniteGP(f, x...)
