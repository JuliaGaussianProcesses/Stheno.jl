"""
    (f_q::GP)(X::AbstractMatrix)

A GP evaluated at `X` is a finite-dimensional GP (i.e. multivariate Normal) whose mean and
covariance are specified by `mean(f_q, X)` and `cov(f_q, X)`.
"""
(f_q::GP)(X::AVM) = GP(f_q, X)
μ_p′(f_q::GP, X::AVM) = FiniteMean(mean(f_q), X)
k_p′(f_q::GP, X::AVM) = FiniteKernel(kernel(f_q), X)
k_p′p(f_p::GP, f_q::GP, X::AVM) = LhsFiniteCrossKernel(kernel(f_q, f_p), X)
k_pp′(f_p::GP, f_q::GP, X′::AVM) = RhsFiniteCrossKernel(kernel(f_p, f_q), X′)
length(::GP, X::AVM) = length(X)
