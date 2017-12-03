"""
    +(z::Real, f::GP)

Return the process which results from adding constant `z` to `GP` `f` from the left.
"""
+(z::Real, f::GP) = GP(+, z, f)
μ_p′(::typeof(+), z::Real, f::GP) = (μ = mean(f); x->z + μ(x))
k_p′(::typeof(+), ::Real, f::GP) = k(f)
k_pp′(f_p::GP, ::typeof(+), ::Real, f::GP) = k(f_p, f)
k_p′p(f_p::GP, ::typeof(+), ::Real, f::GP) = k(f, f_p)

"""
    +(f::GP, z::Real)

Return the process which results from adding constant `z` to `GP` `f` from the right.
"""
+(f::GP, z::Real) = GP(+, f, z)
μ_p′(::typeof(+), f::GP, z::Real) = (μ = mean(f); x->μ(x) + z)
k_p′(::typeof(+), f::GP, ::Real) = k(f)
k_pp′(f_p::GP, ::typeof(+), f::GP, ::Real) = k(f_p, f)
k_p′p(f_p::GP, ::typeof(+), f::GP, ::Real) = k(f, f_p)

"""
    +(g::Function, f::GP)

Return the process which results from adding the deterministic function `g` to `GP` `f` from
the left.
"""
+(g::Function, f::GP) = GP(+, g, f)
μ_p′(::typeof(+), g::Function, f::GP) = (μ = mean(f); x->g(x) + μ(x))
k_p′(::typeof(+), ::Function, f::GP) = k(f)
k_pp′(f_p::GP, ::typeof(+), ::Function, f::GP) = k(f_p, f)
k_p′p(f_p::GP, ::typeof(+), ::Function, f::GP) = k(f, f_p)

"""
    +(f::GP, g::Function)

Return the process which results from adding the deterministic function `g` to `GP` `f` from
the right.
"""
+(f::GP, g::Function) = GP(+, f, g)
μ_p′(::typeof(+), f::GP, g::Function) = (μ = mean(f); x->μ(x) + g(x))
k_p′(::typeof(+), f::GP, ::Function) = k(f)
k_pp′(f_p::GP, ::typeof(+), f::GP, ::Function) = k(f_p, f)
k_p′p(f_p::GP, ::typeof(+), f::GP, ::Function) = k(f, f_p)

"""
    +(f_a::GP, f_b::GP)

Return the process which results from summing `GP`s `fa` and `fb`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
μ_p′(::typeof(+), fa::GP, fb::GP) = (μa = mean(fa); μb = mean(fb); x->μa(x) + μb(x))
k_p′(::typeof(+), fa::GP, fb::GP) = k(fa) + k(fb) + 2 * k(fa, fb)
k_pp′(fp::GP, ::typeof(+), fa::GP, fb::GP) = k(fp, fa) + k(fp, fb)
k_p′p(fp::GP, ::typeof(+), fa::GP, fb::GP) = k(fa, fp) + k(fb, fp)
