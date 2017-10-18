import Base: +

"""
    get_check_gpc(args...)

Check that each elements of `args` which is a `GP` uses the same `GPCollection` object and
return this `GPCollection`.
"""
function get_check_gpc(args...)
    id = findfirst(map(arg->arg isa GP, args))
    gpc = args[id].joint
    @assert all([!(arg isa GP) || arg.joint == gpc for arg in args])
    return gpc
end

"""
    update_gpc(op, args...)

Add a new GP to the collection found somewhere in `args` through the application of `op` to
`args`.
"""
function update_gpc(op, args...)

    # Enforce consistency and get GPCollection.
    gpc = get_check_gpc(op, args...)

    # Compute new mean function.
    μ = μ_p′(op, args...)

    # Compute cross and marginal covariance functions.
    ks = Vector{Any}(length(gpc) + 1)
    for n in 1:length(gpc)
        ks[n] = (k_pp′(op, gpc[n], args...), k_p′p(op, gpc[n], args...))
    end
    ks[end] = k_p′(op, args...)

    # Update the GPCollection and return the new GP.
    push!(gpc, μ, ks)
    return GP(gpc, length(gpc))
end

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate Normal distribution whos mean and covariance are
fully specified by `x` and the mean and covariance functions of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = update_gpc(f_q, x)

function μ_p′(f_q::GP, x::ColOrRowVec)
    μ_q = mean(f_q)
    return n::Int->μ_q(x[n])
end
function k_p′(f_q::GP, x::ColOrRowVec)
    k_q = kernel(f_q)
    return (m::Int, n::Int)->k_q(x[m], x[n])
end
function k_pp′(f_q::GP, f_p::GP, x′::ColOrRowVec)
    k_pq = kernel(f_p, f_q)
    return (x, n′::Int)->k_pq(x, x′[n′])
end
function k_p′p(f_q::GP, f_p::GP, x::ColOrRowVec)
    k_qp = kernel(f_q, f_p)
    return (n::Int, x′)->k_qp(x[n], x′)
end

"""
    +(f_a::GP, f_b::GP)

Return the process which results from summing `GP`s `f_a` and `f_b`.
"""
+(f_a::GP, f_b::GP) = update_gpc(+, f_a, f_b)

function μ_p′(::typeof(+), f_a::GP, f_b::GP)
    μ_a, μ_b = mean(f_a), mean(f_b)
    return x->μ_a(x) + μ_b(x)
end
k_p′(::typeof(+), f_a::GP, f_b::GP) = kernel(f_a) + kernel(f_b) + 2 * kernel(f_a, f_b)
k_pp′(::typeof(+), f_p::GP, f_a::GP, f_b::GP) = kernel(f_p, f_a) + kernel(f_p, f_b)
k_p′p(::typeof(+), f_p::GP, f_a::GP, f_b::GP) = kernel(f_a, f_p) + kernel(f_b, f_p)

