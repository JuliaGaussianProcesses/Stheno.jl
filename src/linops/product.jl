"""
    *(f::GP, g::Union{Real, Function})
    *(f::Union{Real, Function}, g::GP)

Construct the process resulting from the element-wise product of `f` and `g`.
"""
*(f::Union{Real, Function}, g::GP) = GP(*, f, g)
*(f::GP, g::Union{Real, Function}) = GP(*, f, g)

μ_p′(::typeof(*), f::SthenoType, g::SthenoType) = mean(f) * mean(g)

k_p′(::typeof(*), f::Union{Real, Function}, g::GP) = f * k(g) * f
k_p′(::typeof(*), f::GP, g::Union{Real, Function}) = g * k(f) * g

k_pp′(f_p::GP, ::typeof(*), f::GP, g::Union{Real, Function}) = k(f_p, f) * g
k_pp′(f_p::GP, ::typeof(*), f::Union{Real, Function}, g::GP) = k(f_p, g) * f

k_p′p(f_p::GP, ::typeof(*), f::GP, g::Union{Real, Function}) = g * k(f, f_p)
k_p′p(f_p::GP, ::typeof(*), f::Union{Real, Function}, g::GP) = f * k(g, f_p)

# """
#     *(z::Real, f::GP)

# Return the process which results from scaling the `GP` `f` by the real number `z`.
# """
# k_p′(::typeof(*), z::Real, f::GP) = z^2 * k(f)
# k_pp′(f_p::GP, ::typeof(*), z::Real, f::GP) = k(f_p, f) * z
# k_p′p(f_p::GP, ::typeof(*), z::Real, f::GP) = z * k(f, f_p)

# """
#     *(f::GP, g::Union{Real, Function})

# The process (f * g)(x) = f(x) * g(x) if `g` is a function, otherwise (f * g)(x) = f(x) * g.
# """
# k_p′(::typeof(*), f::GP, g::Function) = g * k(f) * g
# k_pp′(f_p::GP, ::typeof(*), f::GP, g::Function) = kernel(f_p, f) * g
# k_p′p(f_p::GP, ::typeof(*), f::GP, g::Function) = g * k(f, f_p)

# """
#     *(f::GP, g::Function)

# The process (g * f)(x) = g(x) * f(x)
# """
# k_p′(::typeof(*), g::Function, f::GP) = g * k(f) * g
# k_pp′(f_p::GP, ::typeof(*), g::Function, f::GP) = kernel(f_p, f) * g
# k_p′p(f_p::GP, ::typeof(*), g::Function, f::GP) = g * k(f, f_p)
