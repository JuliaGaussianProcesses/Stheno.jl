import Base: *

"""
    *(f::Real, g::GP)

Multiplication of a GP `g` from the left by a scalar `f`.
"""
*(f::Real, g::GP) = GP(*, ConstantMean(f), g)
*(f::Function, g::GP) = GP(*, CustomMean(f), g)
μ_p′(::typeof(*), f::MeanFunction, g::GP) = f * mean(g)
k_p′(::typeof(*), f::MeanFunction, g::GP) = OuterKernel(f, kernel(g))
k_pp′(f_p::GP, ::typeof(*), f::MeanFunction, g::GP) = kernel(f_p, g) * f
k_p′p(::typeof(*), f::MeanFunction, g::GP, f_p::GP) = f * kernel(g, f_p)

"""
    *(g::GP, f::Real)

Multiplication of a GP `g` from the right by a scalar `f`.
"""
*(g::GP, f::Real) = GP(*, g, ConstantMean(f))
*(g::GP, f::Function) = GP(*, g, CustomMean(f))
μ_p′(::typeof(*), g::GP, f::MeanFunction) = mean(g) * f
k_p′(::typeof(*), g::GP, f::MeanFunction) = OuterKernel(f, kernel(g))
k_pp′(f_p::GP, ::typeof(*), g::GP, f::MeanFunction) = kernel(f_p, g) * f
k_p′p(::typeof(*), g::GP, f::MeanFunction, f_p::GP) = f * kernel(g, f_p)
