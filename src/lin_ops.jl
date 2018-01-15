import Base: +, *, |
export ←, |

const k = kernel
const CData = ConditionalData

"""
    (f_q::GP)(x::ColOrRowVec)

A GP evaluated at `x` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `x` and the mean function and kernel of `f_q`.

    (f_q::GP{<:Finite})(x::ColOrRowVec)

A GP on a finite-dimensional index set (i.e. a multivariate Normal) indexed at locations `x`
is a new GP with finite-dimensional index set whose cardinality is at most that of `f_q`.
"""
(f_q::GP)(x::ColOrRowVec) = GP(f_q, x)
μ_p′(f_q::GP, x::ColOrRowVec) = FiniteMean(mean(f_q), x)
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

# Some syntactic sugar for conditioning.
struct Assignment
    f::GP
    y::Vector
end
←(f, y) = Assignment(f, y)

"""
    |(f::GP, c::Union{Assignment, Vector{Assignment}})

`|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
returns the conditional (posterior) distribution over everything on the left given the
`Assignment`(s) on the right.
"""
|(f::GP, c::Assignment) = f | [c]
function |(f::GP, c::Vector{Assignment})
    f_obs = [c̄.f for c̄ in c]
    return GP(|, f, f_obs, [c̄.y for c̄ in c], CData(chol(cov(f_obs))))
end

# function μ_p′(::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData)
#     μ, k_ff = mean.(f), Vector{Kernel}(k.(f_obs, f))
#     α = A_ldiv_B!(data.U, At_ldiv_B!(data.U, vcat(f̂...) .- mean_vector(f_obs)))
#     return function(x::Number)
#         kfs = [k isa LhsFinite ? Finite(k, [x]) : Finite(k.k, k.x, [k.y[x]]) for k in k_ff]
#         return μ(x) + dot(reshape(cov(reshape(kfs, :, 1)), :), α)
#     end
# end

μ_p′(::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    ConditionalMean(mean(f), k.(f_obs, f), vcat(f̂...) .- mean_vector(f_obs), data)
k_p′(::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f), k.(f_obs, f), k.(f_obs, f), data)
k_p′p(f_p::GP, ::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f, f_p), k.(f_obs, f), k.(f_obs, f_p), data)
k_pp′(f_p::GP, ::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f_p, f), k.(f_obs, f_p), k.(f_obs, f), data)
dims(::typeof(|), f::GP, ::GP, f̂::Vector) = dims(f)
