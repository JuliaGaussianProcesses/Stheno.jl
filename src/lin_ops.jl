import Base: +, *, |
export ←, |

const μ = mean_function
const k = kernel

"""
    (f_q::GP)(X::AbstractMatrix)

A GP evaluated at `X` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `X` and the mean function and kernel of `f_q`.

    (f_q::GP{<:Finite})(X::AbstractMatrix)

A GP on a finite-dimensional index set (i.e. a multivariate Normal) indexed at locations `X`
is a new GP with finite-dimensional index set whose cardinality is at most that of `f_q`.

Note that nested indexing is currently not supported.
"""
function (f_q::GP)(X::AM)
    isfinite(f_q) && throw(error("Nested indexing not currently supported."))
    return GP(f_q, X)
end
μ_p′(f_q::GP, X::AM) = FiniteMean(μ(f_q), X)
k_p′(f_q::GP, X::AM) = FiniteKernel(k(f_q), X)
function k_p′p(f_p::GP, f_q::GP, X::AM)
    kqp = k(f_q, f_p)
    return isfinite(f_p) ?
        FiniteCrossKernel(kqp.k, X, kqp.X) :
        LhsFiniteCrossKernel(kqp, X)
end
function k_pp′(f_p::GP, f_q::GP, X::AM)
    kpq = kernel(f_p, f_q)
    return isfinite(f_p) ?
        FiniteCrossKernel(kpq.k, kpq.X, X) :
        RhsFiniteCrossKernel(kpq, X)
end
length(::GP, X::AM) = length(X)

"""
    |(f_cond::GP, c::Observation...)

`|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
returns the conditional (posterior) distribution over everything on the left given the
`Observation`(s) on the right.
"""
function |(f_cond::GP, c::Tuple{Vararg{Observation}})
    f, y = getfield.(c, :f), getfield.(c, :y)
    μf, kff = CatMean(μ.(f)), CatKernel(k.(f), k.(f, permutedims(f)))
    return GP(|, f_cond, f, y, μf, kff)
end
μ_p′(::typeof(|), f::GP, f_obs::Tuple{Vararg{<:GP}}, y::Tuple{Vararg{<:AV}}) =
    ConditionalMean()
μ_p′(::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    ConditionalMean(mean(f), k.(f_obs, (f,)), vcat(f̂...) .- mean_vector(f_obs), data)
k_p′(::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f), k.(f_obs, (f,)), k.(f_obs, (f,)), data)
k_p′p(f_p::GP, ::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f, f_p), k.(f_obs, (f,)), k.(f_obs, (f_p,)), data)
k_pp′(f_p::GP, ::typeof(|), f::GP, f_obs::Vector{<:GP}, f̂::Vector{<:Vector}, data::CData) =
    Conditional(k(f_p, f), k.(f_obs, (f_p,)), k.(f_obs, (f,)), data)
length(::typeof(|), f::GP, ::GP, f̂::Vector) = length(f)


struct ConditionalMean <: Function
    kff::FiniteKernel
    μf::Function
    f::AbstractVector{<:Real}
    μg::Function
    kgf::CrossKernel
end
