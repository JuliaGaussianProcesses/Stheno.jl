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
