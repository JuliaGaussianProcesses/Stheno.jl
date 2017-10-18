import Base: +

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate Normal distribution whos mean and
covariance are fully specified by `x` and the mean and covariance functions of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = instantiate_gp(f_q, x)

function μ_p′(f_q::GP, x::ColOrRowVec)
    μ_q = mean(f_q)
    return n::Int->μ_q(x[n])
end

function k_p′(f_q::GP, x::ColOrRowVec)
    k_q = kernel(f_q)
    return (m::Int, n::Int)->k_q(x[m], x[n])
end

# f_q == f_p′.f should hold.
function k_pp′(f_q::GP, f_p::GP, f_p′::GP)
    k_pq, x_p′ = kernel(f_p, f_q), f_p′.args[1]
    return (x, n′::Int)->k_pq(x, x_p′[n′])
end

# f_q == f_p′.f should hold.
function k_p′p(f_q::GP, f_p′::GP, f_p::GP)
    k_qp, x_p′ = kernel(f_q, f_p), f_p′.args[1]
    return (n::Int, x′)->k_qp(x_p′[n], x′)
end

"""
    +(fa::GP, fb::GP)

Return the process which results from summing `GP`s `fa` and `fb`.
"""
+(fa::GP, fb::GP) = instantiate_gp(+, fa, fb)
function μ_p′(::typeof(+), fa::GP, fb::GP)
    μ_a, μ_b = mean(fa), mean(fb)
    return x->μ_a(x) + μ_b(x)
end
k_p′(::typeof(+), fa::GP, fb::GP) = kernel(fa) + kernel(fb) + 2 * kernel(fa, fb)
k_pp′(::typeof(+), f_p::GP, f_p′::GP) =
    kernel(f_p, f_p′.args[1]) + kernel(f_p, f_p′.args[2])
k_p′p(::typeof(+), f_p′::GP, f_p::GP) =
    kernel(f_p′.args[1], f_p) + kernel(f_p′.args[2], f_p)
