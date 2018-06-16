"""
    (f_q::GP)(X::AbstractVecOrMat)

A GP evaluated at `X` is a finite-dimensional GP (i.e. a multivariate Normal).
"""
(f_q::GP)(X::AVM) = GP(f_q, X)

"""
    (f_q::JointGP)(X::AV{<:AVM})

Index the `c`th component `AbstractGP` of `f_q` at `X[c]`.
"""
(f_q::JointGP)(X::AV{<:AVM}) = JointGP(map((f, x)->f(x), f_q.fs, X))

"""
    (f_q::JointGP)(X::AVM)

Index each `c`th component `AbstractGP` of `f_q` at `X`.
"""
(f_q::JointGP)(X::AVM) = JointGP([f(X) for f in f_q])

μ_p′(f_q::GP, X::AVM) = FiniteMean(mean(f_q), X)
k_p′(f_q::GP, X::AVM) = FiniteKernel(kernel(f_q), X)
k_p′p(f_q::GP, X::AVM, f_p::GP) = LhsFiniteCrossKernel(kernel(f_q, f_p), X)
k_pp′(f_p::GP, f_q::GP, X′::AVM) = RhsFiniteCrossKernel(kernel(f_p, f_q), X′)
