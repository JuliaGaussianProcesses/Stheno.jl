export project

"""


"""
project(ϕ, f::GP, g) = GP(project, ϕ, f, g)
μ_p′(::typeof(project), ϕ, f, g) = 
k_p′(::typeof(project), ϕ, f, g) = DegenerateKernel(f, ϕ)
k_pp′(fp::GP, ::typeof(project), ϕ, f::GP, g) = DegenerateCrossKernel()






import Base: +

"""
    +(fa::GP, fb::GP)

Produces a GP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
μ_p′(::typeof(+), fa, fb) = CompositeMean(+, mean(fa), mean(fb))
k_p′(::typeof(+), fa, fb) =
    CompositeKernel(+, kernel(fa), kernel(fb), kernel(fa, fb), kernel(fb, fa))
k_pp′(fp::GP, ::typeof(+), fa, fb) = CompositeCrossKernel(+, kernel(fp, fa), kernel(fp, fb))
k_p′p(::typeof(+), fa, fb, fp::GP) = CompositeCrossKernel(+, kernel(fa, fp), kernel(fb, fp))
