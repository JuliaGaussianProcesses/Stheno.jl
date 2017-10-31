import Base: +
export posterior

const k = kernel

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `x` and the mean function and kernel of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = GP(f_q, x)
function μ_p′(f_q::GP, x::ColOrRowVec)
    μ_q = mean(f_q)
    return n::Int->μ_q(x[n])
end
k_p′(f_q::GP, x::ColOrRowVec) = FullFinite(k(f_q), x)
k_p′p(f_q::GP, f_p′::GP, f_p::GP) = LeftFinite(k(f_q, f_p), f_p′.args[1])
k_pp′(f_q::GP, f_p::GP, f_p′::GP) = RightFinite(k(f_p, f_q), f_p′.args[1])
dims(::GP, x::ColOrRowVec) = length(x)

"""
    posterior(f::GP, f_obs::GP, f̂::ColOrRowVec)

Compute the posterior of the process `f` have observed that `f_obs` equals `f̂`.
"""
posterior(f::GP, f_obs::GP, f̂::ColOrRowVec) =
    GP(posterior, f, f_obs, f̂, PosteriorData(chol(cov(f_obs))))
posterior(f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:ColOrRowVec}) =
    GP(posterior, f, f_obs, f̂, PosteriorData(chol(cov(f_obs))))

function μ_p′(::typeof(posterior), f::GP, f_obs::GP, f̂::ColOrRowVec, data::PosteriorData)
    μ, k_ff̂ = mean(f), k(f, f_obs)
    α = A_ldiv_B!(data.U, At_ldiv_B!(data.U, f̂ .- mean_vector(f_obs)))
    return x::Number->μ(x) + RowVector(broadcast!(k_ff̂, data.tmp, x, data.idx)) * α
end
k_p′(::typeof(posterior), f::GP, f_obs::GP, f̂::ColOrRowVec, data::PosteriorData) =
    PosteriorKernel(k(f), k(f_obs, f), k(f_obs, f), data)
function k_p′p(::typeof(posterior), f_p′::GP, f_p::GP)
    f, f_obs = f_p′.args[1], f_p′.args[2]
    return PosteriorKernel(k(f, f_p), k(f_obs, f), k(f_obs, f_p), k(f_p′).data)
end
function k_pp′(::typeof(posterior), f_p::GP, f_p′::GP)
    f, f_obs = f_p′.args[1], f_p′.args[2]
    return PosteriorKernel(k(f_p, f), k(f_obs, f_p), k(f_obs, f), k(f_p′).data)
end
dims(::typeof(posterior), f::GP, ::GP, f̂::ColOrRowVec) = dims(f)

"""
    +(fa::GP, fb::GP)

Return the process which results from summing `GP`s `fa` and `fb`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
function μ_p′(::typeof(+), fa::GP, fb::GP)
    μ_a, μ_b = mean(fa), mean(fb)
    return x->μ_a(x) + μ_b(x)
end
k_p′(::typeof(+), fa::GP, fb::GP) = k(fa) + k(fb) + 2 * k(fa, fb)
k_pp′(::typeof(+), f_p::GP, f_p′::GP) = k(f_p, f_p′.args[1]) + k(f_p, f_p′.args[2])
k_p′p(::typeof(+), f_p′::GP, f_p::GP) = k(f_p′.args[1], f_p) + k(f_p′.args[2], f_p)
