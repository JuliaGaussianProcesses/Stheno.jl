import Base: cov
export cov, cov!

"""
    cov!(K::CovMat, k::Function, x::T, y::T) where T<:Union{AbstractVector, RowVector}
    cov!(K::CovMat, k::Function, X::T, Y::T) where T<:AbstractMatrix

Compute the covariance matrix whose (i,j)th element is `K[i, j] = k(x[i], y[j])`. If `X` and
`Y` are matrices, then `K[i, j] = k(view(X[i, :]), view(Y[j, :]))`.
"""
cov!(K::CovMat, k::Function, x::T, y::T) where T<:AVector = broadcast!(k, K, x', y)
function cov!(K::CovMat, k::Function, X::T, Y::T) where T<:AbstractMatrix
    for q in 1:size(Y, 2)
        y = view(Y, :, q)
        for p in 1:size(X, 2)
            K[q, p] = k(view(X, :, p), y)
        end
    end
    return K
end

"""
    cov(k::Function, x::T, y::T) where T<:Union{AbstractVector, RowVector}
    cov(k::Function, X::T, y::T) where T<:AbstractMatrix

Allocate memory for the covariance matrix and call `cov!`.
"""
cov(k::Function, x::T, y::T) where T<:AVector =
    cov!(CovMat(length(x), length(y)), k, x, y)
cov(k::Function, X::T, Y::T) where T<:AbstractMatrix =
    cov!(CovMat(size(X, 2), size(Y, 2)), k, X, Y)

"""
    EQ()

Returns the Exponentiated Quadratic kernel, which has no free parameters.
"""
EQ() = _EQ
_EQ(x::T, y::T) where T<:Union{Real, AbstractVector{<:Real}} = exp(-0.5 * sum(abs2, x - y))
uses_sq_dist(::typeof(_EQ)) = true

"""
    RQ(α)

Generate a Rational Quadratic kernel whose kurtosis is `α`.
"""
function RQ(α)
    k = function (x::T, y::T) where T<:Union{Real, AbstractVector{<:Real}}
        return (1.0 - 0.5 * sum(abs2, x - y))^α
    end
    global (uses_sq_dist(::typeof(k)) = true)
    return k
end

# TODO:
# Efficient kernel evaluation in high-dimensions via matrix-multiplication.

# TODO:
# 1) How can we enable kernel-based dispatch for differently parametrised kernels? Can we
# enquire regarding their traits or something / create a function that returns a boolean
# indicating properties that a particular function satisfies?
# 2) Implement Linear, Matern

# TODO:
# Input transformations: periodic, linear transforms etc. What about invariance-inducing
# transformations?

