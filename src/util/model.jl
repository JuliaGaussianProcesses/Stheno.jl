macro model(expr)

    # Check it's a function and deconstruct.
    foo = splitdef(expr)

    # Transform provided expression to use a GPC with each GP.
    foo[:body] = Expr(:block,
        :(gpc = Stheno.GPC()),
        postwalk(x->@capture(x, GP(xs__)) ? :(Stheno.wrap(GP($(xs...)), gpc)) : x, foo[:body]),
    )

    # Recombine into function expression.
    return esc(combinedef(foo))
end
