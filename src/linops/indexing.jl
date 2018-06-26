"""
    (f_q::GP)(X::AbstractDataSet)

A GP evaluated at `X` is a finite-dimensional GP (i.e. a multivariate Normal).
"""
(f_q::GP)(X::AbstractDataSet) = GP(f_q, X)

"""
    (f_q::JointGP)(X::BlockData)

Index the `c`th component `AbstractGP` of `f_q` at `X[c]`.
"""
(f_q::JointGP)(X::BlockData) = JointGP(map((f, x)->f(x), f_q.fs, blocks(X)))

"""
    (f_q::JointGP)(X::AbstractDataSet)

Index each `c`th component `AbstractGP` of `f_q` at `X`.
"""
(f_q::JointGP)(X::ADS) = JointGP([f(X) for f in f_q])

μ_p′(f_q::GP, X::AVM) = FiniteMean(mean(f_q), X)
k_p′(f_q::GP, X::AVM) = FiniteKernel(kernel(f_q), X)
k_p′p(f_q::GP, X::AVM, f_p::GP) = LhsFiniteCrossKernel(kernel(f_q, f_p), X)
k_pp′(f_p::GP, f_q::GP, X′::AVM) = RhsFiniteCrossKernel(kernel(f_p, f_q), X′)

# Sugar. This behaviour needs to be formalised and isolated at some point.
(f_q::GP)(X::AVM) = f_q(ADS(X))
(f_q::JointGP)(X::AVM) = f_q(ADS(X))
