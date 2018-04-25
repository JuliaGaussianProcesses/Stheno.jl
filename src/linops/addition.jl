import Base: +

"""
    +(fa::GP, fb::GP)

Produces a GP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
μ_p′(::typeof(+), fa, fb) = CompositeMean(+, mean(fa), mean(fb))
k_p′(::typeof(+), fa, fb) = CompositeKernel(+, kernel(fa), kernel(fb), kernel(fa, fb), kernel(fb, fa))
k_pp′(fp::GP, ::typeof(+), fa, fb) = CompositeCrossKernel(+, kernel(fp, fa), kernel(fp, fb))
k_p′p(fp::GP, ::typeof(+), fa, fb) = CompositeCrossKernel(+, kernel(fa, fp), kernel(fb, fp))
