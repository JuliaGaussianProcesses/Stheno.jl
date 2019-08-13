import Base: +, -

"""
    +(fa::AbstractGP, fb::AbstractGP)

Produces an AbstractGP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
function +(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    return CompositeGP((+, fa, fb), fa.gpc)
end

const add_args = Tuple{AbstractGP, AbstractGP}

mean_vector((fa, fb)::add_args, x::AV) = mean_vector(fa, x) .+ mean_vector(fb, x)
function cov_mat((fa, fb)::add_args, x::AV)
    return cov_mat(fa, x) .+ cov_mat(fb, x) .+ xcov_mat(fa, fb, x) .+ xcov_mat(fb, fa, x)
end
function cov_mat((fa, fb)::add_args, x::AV, x′::AV)
    C_a, C_b = cov_mat(fa, x, x′), cov_mat(fb, x, x′)
    return C_a .+ C_b .+ xcov_mat(fa, fb, x, x′) .+ xcov_mat(fb, fa, x, x′)
end
function diag_cov_mat((fa, fb)::add_args, x::AV)
    C_a, C_b = cov_mat_diag(fa, x), cov_mat_diag(fb, x)
    return C_a .+ C_b .+ cov_mat_diag(fa, fb, x) .+ cov_mat_diag(fb, fa, x)
end
# NEED TO BE ABLE TO GET THE DIAGONAL OF AN XCOV APPARENTLY!



μ_p′(::typeof(+), fa, fb) = mean(fa) + mean(fb)
function k_p′(::typeof(+), fa, fb)
    return kernel(fa) + kernel(fb) + kernel(fa, fb) + kernel(fb, fa)
    # k_out = kernel(fa) + kernel(fb) + kernel(fa, fb) + kernel(fb, fa)
    # if k_out isa CompositeCrossKernel
    #     return CompositeKernel(+, k_out.x...)
    # else
    #     return k_out
    # end
end
k_pp′(fp::GP, ::typeof(+), fa, fb) = kernel(fp, fa) + kernel(fp, fb)
k_p′p(::typeof(+), fa, fb, fp::GP) = kernel(fa, fp) + kernel(fb, fp)

"""
    +(c, f::GP)
    +(f::GP, c)

Adding a deterministic quantity to a GP just shifts the mean.
"""
+(c, f::GP) = GP(c, ZeroKernel(), f.gpc) + f
+(f::GP, c) = f + GP(c, ZeroKernel(), f.gpc)

# Define negation in terms of other operations.
-(f::GP) = -1 * f
-(f::GP, c) = f + (-c)
-(c, f::GP) = c + (-f)
-(f::GP, g::GP) = f + (-g)
