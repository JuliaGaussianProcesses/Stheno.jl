"""
    (f::GP)(x::AbstractVector)

Construct a `FiniteGP` representing the projection of `f` at `x`.
"""
(f::GP)(x...) = FiniteGP(f, x...)
