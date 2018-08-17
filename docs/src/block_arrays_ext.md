# Extensions to BlockArrays.jl

This package has had to implement a number of extensions to `BlockArrays.jl`, which are detailed below. The hope is that these will eventually find their way into `BlockArrays.jl`, as they don't obviously belong in this package.

## Block Symmetric Matrices

Let us define a block-symmetric matrix `X` to be an `AbstractBlockMatrix` endowed with the property `getblock(X, m, n) == transpose(getblock(X, n, m))`, which we call block-symmetry. It is straightforward to see that block-symmetry implies:
1. symmetry in the usual sense (`X[p, q] == X[q, p]`), and
2. `size(getblock(X, m, n)) == size(getblock(X, n, m))`, and
3. `getblock(X, p, p)` is `Symmetric` for all `p`.

It will be useful for our subsequent discussion to define `blocksizes(X, n)` to be a vector whose `p`th element is `size(getblock(X, p, 1), n)`.


### Why bother?

The last property is particularly important for our purposes: it will allow us to guarantee that the Cholesky factorisation of a positive definite block-symmetric matrix can be represented as block matrix for which `blocksizes` is exactly the same.


### Which block matrices can be block symmetric?

It is important to ask which block matrices can potentially to be block-symmetric. A necessary condition for a block matrix to be block symmetric is that `blocksizes(X, 1) == blocksizes(X, 2)`. From an implementation perspective this condition is sufficient: `BlockSymmetric(X)` simply checks that this condition holds, and then enforces that `X` be block symmetric. This is analogous to the way which `Symmetric(Y)`, for some `AbstractMatrix` `Y`, checks that `Y` is square and then enforces `Y[p, q] == Y[q, p]`.


## Types under consideration

The basic types under consideration are:
- `const ABV{T} = AbstractBlockVector{T}`
- `const ABM{T} = AbstractBlockMatrix{T}`
- `const ABMV{T} = Union{ABV{T}, ABM{T}}`
- `const BS{T} = BlockSymmetric{T, <:AbstractBlockMatrix{T}}`

Furthermore, we require the following wrappers around `BlockSymmetric` matrices:
- `UpperTriangular{T, <:BS{T}}`
- `LowerTriangular{T, <:BS{T}}`
- `LazyPDMat{T, <:BS{T}}`

For all of the above, we require "sensible" implementations of the following unary functions:
- `getblock`
- `nblocks`
- `blocksizes`
- `setblock!`
- `copy`
- `transpose` (0.6-specific, changes in 0.7 / 1.0)
- `adjoint` (0.6-specific, changes in 0.7 / 1.0)


