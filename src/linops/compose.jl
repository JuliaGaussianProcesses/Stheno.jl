import Base: ∘
export ∘, select, stretch, periodic

"""
    ∘(f::GP, g)

Constructs the GP f′ given by f′(x) := f(g(x))
"""
∘(f::GP, g) = GP(f.gpc, transform, f, g)

μ_p′(::typeof(transform), f::GP, g) = transform(mean(f), g)
k_p′(::typeof(transform), f::GP, g) = transform(kernel(f), g)
k_pp′(fp::GP, ::typeof(transform), f::GP, g) = transform(kernel(fp, f), g, Val(2))
k_p′p(::typeof(transform), f::GP, g, fp::GP) = transform(kernel(f, fp), g, Val(1))

# Sugar

"""
    select(f::GP, idx)

Select the dimensions of the input to `f` given by `idx`.
"""
select(f::GP, idx) = f ∘ Select(idx)

"""
    stretch(f::GP, l::Real)

Stretch all inputs by amount `l`.
"""
stretch(f::GP, l::Real) = f ∘ Stretch(l)

"""
    stretch(f::GP, A::AbstractVector)

Stretch each input to `f` by the amount specified in `a`. 
"""
stretch(f::GP, a::AbstractVector) = stretch(f, Diagonal(a))

"""
    stretch(f::GP, A::AbstractMatrix)

Multiple the inputs to `f` by `A`.
"""
stretch(f::GP, A::AbstractMatrix) = f ∘ LinearTransform(A)

"""
    periodic(g::GP, f::Real)

Produce a GP with period `f`.
"""
periodic(g::GP, f::Real) = g ∘ Periodic(f)
