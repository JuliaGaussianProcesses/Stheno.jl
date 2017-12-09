import Base: +, *, |
export posterior, ←, |

const k = kernel

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `x` and the mean function and kernel of `f_q`.

    (f_q::GP{<:Finite})(x::ColOrRowVec)

A GP on a finite-dimensional index set (i.e. a multivariate Normal) indexed at locations `x`
is a new GP with finite-dimensional index set whose cardinality is at most that of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = GP(f_q, x)

function μ_p′(f_q::GP, x::ColOrRowVec)
    μ_q = mean(f_q)
    return n::Int->μ_q(x[n])
end
k_p′(f_q::GP, x::ColOrRowVec) = Finite(k(f_q), x)
k_p′p(f_p::GP, f_q::GP, x::ColOrRowVec) =
    isfinite(f_p) ?
        Finite(k(f_q, f_p), x, 1:size(k(f_p), 2)) :
        LhsFinite(k(f_q, f_p), x)
k_pp′(f_p::GP, f_q::GP, x::ColOrRowVec) =
    isfinite(f_p) ?
        Finite(k(f_p, f_q), 1:size(k(f_p), 1), x) :
        RhsFinite(k(f_p, f_q), x)
dims(::GP, x::ColOrRowVec) = length(x)

"""
    posterior(f::GP, f_obs::GP, f̂::Vector)

Compute the posterior of the process `f` have observed that `f_obs` equals `f̂`.
"""
posterior(f::GP, f_obs::GP, f̂::Vector) =
    GP(posterior, f, f_obs, f̂, ConditionalData(chol(cov(f_obs))))
posterior(f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}) =
    GP(posterior, f, f_obs, f̂, ConditionalData(chol(cov(f_obs))))

function μ_p′(::typeof(posterior), f::GP, f_obs::GP, f̂::Vector, data::ConditionalData)
    μ, k_ff̂ = mean(f), k(f, f_obs)
    α = A_ldiv_B!(data.U, At_ldiv_B!(data.U, f̂ .- mean_vector(f_obs)))
    return x::Number->μ(x) + RowVector(broadcast!(k_ff̂, data.tmp, x, data.idx)) * α
end
k_p′(::typeof(posterior), f::GP, f_obs::GP, f̂::Vector, data::ConditionalData) =
    Conditional(k(f), k(f_obs, f), k(f_obs, f), data)
k_p′p(f_p::GP, ::typeof(posterior), f::GP, f_obs::GP, f̂::Vector, data::ConditionalData) =
    Conditional(k(f, f_p), k(f_obs, f), k(f_obs, f_p), data)
k_pp′(f_p::GP, ::typeof(posterior), f::GP, f_obs::GP, f̂::Vector, data::ConditionalData) =
    Conditional(k(f_p, f), k(f_obs, f_p), k(f_obs, f), data)
dims(::typeof(posterior), f::GP, ::GP, f̂::Vector) = dims(f)

# Some syntactic sugar for conditioning.
struct Assignment
    f::GP
    y::Vector
end
←(f, y) = Assignment(f, y)
|(f::GP, c::Assignment) = posterior(f, c.f, c.y)
|(f::GP, c::Vector{Assignment}) = posterior(f, [c̄.f for c̄ in c], [c̄.y for c̄ in c])

