import Base: +
export posterior

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate Normal distribution whos mean and
covariance are fully specified by `x` and the mean and covariance functions of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = Normal(f_q, x)

function μ_p′(f_q::GP, x::ColOrRowVec)
    μ_q = mean(f_q)
    return n::Int->μ_q(x[n])
end
k_p′(f_q::GP, x::ColOrRowVec) = Finite(kernel(f_q), x)
k_p′p(f_q::GP, f_p′::Normal, f_p::AbstractGP) = LeftFinite(kernel(f_q, f_p), f_p′.args[1])
k_pp′(f_q::GP, f_p::AbstractGP, f_p′::Normal) = RightFinite(kernel(f_p, f_q), f_p′.args[1])
dims(::GP, x::ColOrRowVec) = length(x)

"""
    posterior(f::AbstractGP, f_obs::Normal, f̂::ColOrRowVec)

Compute the posterior of the process `f` have observed that `f_obs` equals `f̂`.
"""
posterior(f::T, f_obs::Normal, f̂::ColOrRowVec) where T<:AbstractGP =
    T(posterior, f, f_obs, f̂)
posterior(f::T, f_obs::Vector{Normal}, f̂::Vector{ColOrRowVec}) where T<:AbstractGP =
    T(posterior, f, f_bs, f̂)

dims(::typeof(posterior), f::Normal, ::Normal, f̂::ColOrRowVec) = dims(f)

function μ_p′(::typeof(posterior), f::AbstractGP, f_obs::Normal, f̂::ColOrRowVec)
    idx = eachindex(f_obs)
    μ, k_ff̂, U = mean(f), kernel(f, f_obs), chol(cov(f_obs))
    α = (U \ At_ldiv_B(U, f̂ .- mean(f_obs).(idx)))
    tmp = Vector{Float64}(length(idx))
    return x::Number->μ(x) + RowVector(broadcast!(k_ff̂, tmp, x, idx)) * α
end
# function μ_p′(
#     ::typeof(posterior),
#     f::AbstractGP,
#     f_obs::Vector{Normal}, 
#     f̂::Vector{ColOrRowVec},
# )
    
# end

function k_p′(::typeof(posterior), f::AbstractGP, f_obs::Normal, f̂::ColOrRowVec)
    k_f, k_f̂f = kernel(f), kernel(f_obs, f)
    idx, U = collect(eachindex(f_obs)), chol(cov(f_obs))
    tmp, tmp′ = Vector{Float64}(length(idx)), Vector{Float64}(length(idx))
    return function(x, x′)
        a = At_ldiv_B!(U, broadcast!(k_f̂f, tmp, idx, x))
        b = At_ldiv_B!(U, broadcast!(k_f̂f, tmp′, idx, x′))
        return k_f(x, x′) - dot(a, b)
    end
end

function k_pp′(::typeof(posterior), f_p::AbstractGP, f_p′::AbstractGP)
    f, f_obs = f_p′.args[1], f_p′.args[2]
    k_ff′, k_f̂f, k_f̂f′ = kernel(f_p, f), kernel(f_obs, f_p), kernel(f_obs, f)
    idx, U = collect(eachindex(f_obs)), chol(cov(f_obs))
    tmp, tmp′ = Vector{Float64}(length(idx)), Vector{Float64}(length(idx))
    return function(x, x′)
        a = At_ldiv_B!(U, broadcast!(k_f̂f, tmp, idx, x))
        b = At_ldiv_B!(U, broadcast!(k_f̂f′, tmp′, idx, x′))
        return k_ff′(x, x′) - dot(a, b)
    end
end

function k_p′p(::typeof(posterior), f_p′::AbstractGP, f_p::AbstractGP)
    f, f_obs = f_p′.args[1], f_p′.args[2]
    k_f′f, k_f̂f′, k_f̂f = kernel(f, f_p), kernel(f_obs, f), kernel(f_obs, f_p)
    idx, U = collect(eachindex(f_obs)), chol(cov(f_obs))
    tmp, tmp′ = Vector{Float64}(length(idx)), Vector{Float64}(length(idx))
    return function(x, x′)
        a = At_ldiv_B!(U, broadcast!(k_f̂f′, tmp, idx, x))
        b = At_ldiv_B!(U, broadcast!(k_f̂f, tmp′, idx, x′))
        return k_f′f(x, x′) - dot(a, b)
    end
end

"""
    +(fa::GP, fb::GP)

Return the process which results from summing `GP`s `fa` and `fb`.
"""
+(fa::GP, fb::GP) = GP(+, fa, fb)
function μ_p′(::typeof(+), fa::GP, fb::GP)
    μ_a, μ_b = mean(fa), mean(fb)
    return x->μ_a(x) + μ_b(x)
end
k_p′(::typeof(+), fa::GP, fb::GP) = kernel(fa) + kernel(fb) + 2 * kernel(fa, fb)
k_pp′(::typeof(+), f_p::GP, f_p′::GP) =
    kernel(f_p, f_p′.args[1]) + kernel(f_p, f_p′.args[2])
k_p′p(::typeof(+), f_p′::GP, f_p::GP) =
    kernel(f_p′.args[1], f_p) + kernel(f_p′.args[2], f_p)
