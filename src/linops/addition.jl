"""
    +(fa::GP, fb::GP)

Produces a GP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
μ_p′(::typeof(+), fa, fb) = CompositeMean(+, mean(fa), mean(fb))
k_p′(::typeof(+), fa, fb) = CompositeKernel(+, k(fa), k(fb), k(fa, fb), k(fb, fa))
k_pp′(fp::GP, ::typeof(+), fa, fb) = CompositeCrossKernel(+, k(fp, fa), k(fp, fb))
k_p′p(fp::GP, ::typeof(+), fa, fb) = CompositeCrossKernel(+, k(fa, fp), k(fb, fp))
