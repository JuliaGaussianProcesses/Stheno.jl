import Base: +

"""
    +(fa::GP, fb::GP)

Produces a GP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)

"""
    +(fa::Union{GP, BlockGP}, fb::Union{GP, BlockGP})

Same as above, but for a collection of GPs.
"""
+(fa::BlockGP, fb::BlockGP) = BlockGP(fa.fs .+ fb.fs)
+(fa::BlockGP, fb::GP) = BlockGP(fa.fs .+ fb)
+(fa::GP, fb::BlockGP) = BlockGP(fa .+ fb.fs)

μ_p′(::typeof(+), fa, fb) = mean(fa) + mean(fb)
k_p′(::typeof(+), fa, fb) = kernel(fa) + kernel(fb) + kernel(fa, fb) + kernel(fb, fa)
k_pp′(fp::GP, ::typeof(+), fa, fb) = kernel(fp, fa) + kernel(fp, fb)
k_p′p(::typeof(+), fa, fb, fp::GP) = kernel(fa, fp) + kernel(fb, fp)
