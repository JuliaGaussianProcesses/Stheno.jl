"""
    +(f::GP, g::GP)
    +(f, g::GP)
    +(f::GP, g)

Add together

Return a mean function which computes (f + g)(x) = f(x) + g(x). Clearly only makes sense if
the domain of `f` and `g` are the same. This is, however, not checked.
"""
+(f, g::GP) = GP(+, f, g)
+(f::GP, g) = GP(+, f, g)
+(f::GP, g::GP) = GP(+, f, g)
μ_p′(::typeof(+), f, g) = mean(f) + mean(g)
k_p′(::typeof(+), f, g) = k(f) + k(g) + 2 * k(f, g)
k_pp′(fp::GP, ::typeof(+), fa, fb) = k(fp, fa) + k(fp, fb)
k_p′p(fp::GP, ::typeof(+), fa, fb) = k(fa, fp) + k(fb, fp)
