import Base: *

"""
    *(f, g::GP)

Multiplication of a GP `g` from the left by a number or function.
"""
*(f::MeanFunction, g::GP) = GP(g.gpc, *, f, g)
μ_p′(::typeof(*), f::MeanFunction, g::GP) = f * mean(g)
k_p′(::typeof(*), f::MeanFunction, g::GP) = OuterKernel(f, kernel(g))
k_pp′(f_p::GP, ::typeof(*), f::MeanFunction, g::GP) = kernel(f_p, g) * f
k_p′p(::typeof(*), f::MeanFunction, g::GP, f_p::GP) = f * kernel(g, f_p)

*(f::Real, g::GP) = ConstMean(f) * g
*(f::Function, g::GP) = CustomMean(f) * g

"""
    *(g::GP, f)

Multiplication of a GP `g` from the right by a number or function.
"""
*(g::GP, f::MeanFunction) = GP(g.gpc, *, g, f)
μ_p′(::typeof(*), g::GP, f::MeanFunction) = mean(g) * f
k_p′(::typeof(*), g::GP, f::MeanFunction) = OuterKernel(f, kernel(g))
k_pp′(f_p::GP, ::typeof(*), g::GP, f::MeanFunction) = kernel(f_p, g) * f
k_p′p(::typeof(*), g::GP, f::MeanFunction, f_p::GP) = f * kernel(g, f_p)

*(g::GP, f::Real) = g * ConstMean(f)
*(g::GP, f::Function) = g * CustomMean(f)
*(g::GP, f) = g * CustomMean(f)
