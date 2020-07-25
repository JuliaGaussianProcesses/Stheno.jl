@static if VERSION >= v"1.3"

    """
        (f::AbstractGP)(x::AbstractVector)

    Construct a `FiniteGP` representing the projection of `f` at `x`.
    """
    (f::AbstractGP)(x...) = FiniteGP(f, x...)
else

    """
        (f::Union{GP, CompositeGP})(x::AbstractVector)

    Construct a `FiniteGP` representing the projection of `f` at `x`.
    """
    (f::GP)(x...) = FiniteGP(f, x...)
    (f::CompositeGP)(x...) = FiniteGP(f, x...)
end
