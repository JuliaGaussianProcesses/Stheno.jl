
# # Number of observations.
# @inline nobs(x::AbstractVector) = length(x)
# @inline nobs(X::AbstractMatrix) = size(X, 2)
# @inline nobs(X::AbstractVector{<:AbstractArray}) = sum(nobs, X)

# # Dimensionality of observations.
# @inline nfeatures(x::AbstractVector) = 1
# @inline nfeatures(X::AbstractMatrix) = size(X, 1)

# # Indices of each observation.
# @inline eachobs(X::AbstractVecOrMat) = Base.OneTo(nobs(X))
# @inline eachobs(X::AbstractVector{<:AbstractArray}) = eachobs.(X)

# """
#     getobs(x::AbstractVecOrMat, indices)

# Return the observations corresponding to `indices` from `x`. If all observations are
# requested, then `getobs(x, indices) === x`.
# """
# @inline getobs(x::AbstractVector, n) = n == eachindex(x) ? x : x[n]
# @inline getobs(X::AbstractMatrix, n) = n == 1:nobs(X) ? X : X[:, n]
# @inline getobs(x::AbstractVector, ::Colon) = x
# @inline getobs(X::AbstractMatrix, ::Colon) = X

# # Fallback implementations for `unary_obswise`.
# @inline unary_obswise(f, X::AbstractVecOrMat) = unary_obswise_fallback(f, X)

# unary_obswise_fallback(f, X::AVM) = map(n->f(getobs(X, n)), 1:nobs(X))

# # Fallback implementations for `binary_obswise`.
# @inline binary_obswise(f, X::AbstractVecOrMat) =
#     isstationary(f) ?
#         Fill(f(getobs(X, 1), getobs(X, 1)), nobs(X)) :
#         binary_obswise(f, X, X)
# @inline binary_obswise(f, X::AVM, X′::AVM) = binary_obswise_fallback(f, X, X′)

# binary_obswise_fallback(f, X::AVM, X′::AVM) =
#     map(n->f(getobs(X, n), getobs(X′, n)), 1:nobs(X))

# Can convert to AbstractVector / AbstractMatrix if MeanFunction / Kernel has finite dims.
