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
+(fa::BlockGP, fb::GP) = BlockGP(fa.fs .+ Ref(fb))
+(fa::GP, fb::BlockGP) = BlockGP(Ref(fa) .+ fb.fs)

μ_p′(::typeof(+), fa, fb) = mean(fa) + mean(fb)
function k_p′(::typeof(+), fa, fb)
    k_out = kernel(fa) + kernel(fb) + kernel(fa, fb) + kernel(fb, fa)
    if k_out isa CompositeCrossKernel
        return CompositeKernel(+, k_out.x...)
    else
        return k_out
    end
end
k_pp′(fp::GP, ::typeof(+), fa, fb) = kernel(fp, fa) + kernel(fp, fb)
k_p′p(::typeof(+), fa, fb, fp::GP) = kernel(fa, fp) + kernel(fb, fp)

"""
    +(c::Real, f::GP)
    +(f::GP, c::Real)

Adding a constant to a GP just shifts the mean.
"""
+(c::T, f::GP) where T<:Real = GP(c, ZeroKernel{T}(), f.gpc) + f
+(f::GP, c::T) where T<:Real = f + GP(c, ZeroKernel{T}(), f.gpc)
