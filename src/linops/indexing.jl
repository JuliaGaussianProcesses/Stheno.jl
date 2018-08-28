"""
    (f_q::GP)(X::AbstractVector)

A GP evaluated at `X` is a finite-dimensional GP (i.e. a multivariate Normal).
"""
(f_q::GP)(X::AbstractVector) = GP(f_q, X)

"""
    (f_q::BlockGP)(X::BlockData)

Index the `c`th component `AbstractGP` of `f_q` at `X[c]`.

    (f_q::BlockGP)(X::AbstractVector)

Index each `c`th component `AbstractGP` of `f_q` at `X`.
"""
(f_q::BlockGP)(X::BlockData) = BlockGP(map((f, x)->f(x), f_q.fs, blocks(X)))
(f_q::BlockGP)(X::AbstractVector) = BlockGP([f(X) for f in f_q.fs])

μ_p′(f_q::GP, X::AVM) = finite(mean(f_q), X)
k_p′(f_q::GP, X::AVM) = finite(kernel(f_q), X)
k_p′p(f_q::GP, X::AVM, f_p::GP) = lhsfinite(kernel(f_q, f_p), X)
k_pp′(f_p::GP, f_q::GP, X′::AVM) = rhsfinite(kernel(f_p, f_q), X′)
