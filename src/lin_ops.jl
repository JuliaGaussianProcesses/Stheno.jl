import Base: +

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate Normal distribution whos mean and
covariance are fully specified by `x` and the mean and covariance functions of `f_q`.
"""
function (f_q::GP)(x::ColOrRowVec)
    μ_q, k_q = mean(f_q), kernel(f_q)
    μ_p′ = n::Int->μ_q(x[n])
    k_p′ = (m::Int, n::Int)->k_q(x[m], x[n])
    return GP(f_q, x, μ_p′, k_p′, f_q.k_x)
end

# f_q == f_p′.f should hold.
function k_pp′(f_q::GP, f_p::GP, f_p′::GP)
    k_pq, x_p′ = kernel(f_p, f_q), f_p′.args
    return (x, n′::Int)->k_pq(x, x_p′[n′])
end

# f_q == f_p′.f should hold.
function k_p′p(f_q::GP, f_p′::GP, f_p::GP)
    k_qp, x_p′ = kernel(f_q, f_p), f_p′.args
    return (n::Int, x′)->k_qp(x_p′[n], x′)
end

# """
#     +(f_a::GP, f_b::GP)

# Return the process which results from summing `GP`s `f_a` and `f_b`.
# """
# function +(f_a::GP, f_b::GP)
#     μ_a, μ_b = mean(f_a), mean(f_b)
#     μ_p′ = x->μ_a(x) + μ_b(x)
#     k_p′ = kernel(f_a) + kernel(f_b) + 2 * kernel(f_a, f_b)
#     return BranchGP(+, (f_a, f_b), μ_p′, k_p′, merge!(ObjectIdDict(), f_a.k_x, f_b.k_x))
# end

# k_pp′(::typeof(+), f_p::AbstractGP, f_p′::BranchGP) =
#     kernel(f_p, f_p′.args[1]) + kernel(f_p, f_p′.args[2])
# k_p′p(::typeof(+), f_p′::BranchGP, f_p::AbstractGP) =
#     kernel(f_p′.args[1], f_p) + kernel(f_p′.args[2], f_p)
