import Base: ∘

"""
    ∘(f::AbstractGP, g)

Constructs the GP f′ given by f′(x) := f(g(x))
"""
∘(f::AbstractGP, g) = GP(transform, f, g)

μ_p′(::typeof(transform), f::AbstractGP, g) = transform(mean(f), g)
k_p′(::typeof(transform), f::AbstractGP, g) = transform(kernel(f), g)
k_pp′(fp::GP, ::typeof(transform), f::AbstractGP, g) = transform(kernel(fp, f), g, Val(2))
k_p′p(::typeof(transform), f::AbstractGP, g, fp::GP) = transform(kernel(f, fp), g, Val(1))
