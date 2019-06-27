import Base: ∘

"""
    ∘(f::GP, g)

Constructs the GP f′ given by f′(x) := f(g(x))
"""
∘(f::GP, g) = GP(f.gpc, transform, f, g)

μ_p′(::typeof(transform), f::GP, g) = transform(mean(f), g)
k_p′(::typeof(transform), f::GP, g) = transform(kernel(f), g)
k_pp′(fp::GP, ::typeof(transform), f::GP, g) = transform(kernel(fp, f), g, Val(2))
k_p′p(::typeof(transform), f::GP, g, fp::GP) = transform(kernel(f, fp), g, Val(1))
