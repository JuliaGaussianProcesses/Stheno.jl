"""
    *(f::GP, z::Real)

Return the process which results from scaling the `GP` `f` by the real number `z`.
"""
*(f::GP, z::Real) = GP(*, f, z)
μ_p′(::typeof(*), f::GP, z::Real) = (μ_f = mean(f); x->μ_f(x) * z)
k_p′(::typeof(*), f::GP, z::Real) = k(f) * z^2
k_pp′(f_p::GP, ::typeof(*), f::GP, z::Real) = k(f_p, f) * z
k_p′p(f_p::GP, ::typeof(*), f::GP, z::Real) = z * k(f, f_p)

"""
    *(z::Real, f::GP)

Return the process which results from scaling the `GP` `f` by the real number `z`.
"""
*(z::Real, f::GP) = GP(*, z, f)
μ_p′(::typeof(*), z::Real, f::GP) = (μ_f = mean(f); x->z * μ_f(x))
k_p′(::typeof(*), z::Real, f::GP) = z^2 * k(f)
k_pp′(f_p::GP, ::typeof(*), z::Real, f::GP) = k(f_p, f) * z
k_p′p(f_p::GP, ::typeof(*), z::Real, f::GP) = z * k(f, f_p)

"""
    *(f::GP, g::Function)

The process (f * g)(x) = f(x) * g(x)
"""
*(f::GP, g::Function) = GP(*, f, g)
μ_p′(::typeof(*), f::GP, g::Function) = (μ_f = mean(f); x->μ_f(x) * g(x))
k_p′(::typeof(*), f::GP, g::Function) = g * k(f) * g
k_pp′(f_p::GP, ::typeof(*), f::GP, g::Function) = kernel(f_p, f) * g
k_p′p(f_p::GP, ::typeof(*), f::GP, g::Function) = g * k(f, f_p)

"""
    *(f::GP, g::Function)

The process (g * f)(x) = g(x) * f(x)
"""
*(g::Function, f::GP) = GP(*, g, f)
μ_p′(::typeof(*), g::Function, f::GP) = (μ_f = mean(f); x->g(x) * μ_f(x))
k_p′(::typeof(*), g::Function, f::GP) = g * k(f) * g
k_pp′(f_p::GP, ::typeof(*), g::Function, f::GP) = kernel(f_p, f) * g
k_p′p(f_p::GP, ::typeof(*), g::Function, f::GP) = g * k(f, f_p)
