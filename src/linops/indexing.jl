"""
    (f::GP)(x::AbstractVector)
    (f::BlockGP)(x::BlockGP)

Construct a `FiniteGP` representing the projection of `f` at `x`.
"""
(f::GP)(x...) = FiniteGP(f, x...)
(f::BlockGP)(x...) = FiniteGP(f, x...)
