var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "block_arrays_ext.html#",
    "page": "BlockArrays extensions",
    "title": "BlockArrays extensions",
    "category": "page",
    "text": ""
},

{
    "location": "block_arrays_ext.html#Extensions-to-BlockArrays.jl-1",
    "page": "BlockArrays extensions",
    "title": "Extensions to BlockArrays.jl",
    "category": "section",
    "text": "This package has had to implement a number of extensions to BlockArrays.jl, which are detailed below. The hope is that these will eventually find their way into BlockArrays.jl, as they don\'t obviously belong in this package."
},

{
    "location": "block_arrays_ext.html#BlockSymmetric-1",
    "page": "BlockArrays extensions",
    "title": "BlockSymmetric",
    "category": "section",
    "text": "Let us define a BlockSymmetric <: AbstractBlockMatrix X to be an AbstractMatrix endowed with the property getblock(X, m, n) == transpose(getblock(X, n, m)), which we call \"block symmetry\". It is straightforward to see that block symmetry implies:symmetry in the usual sense (X[p, q] == X[q, p]), and\nsize(getblock(X, m, n)) == size(getblock(X, n, m)), and\ngetblock(X, p, p) is Symmetric for all p.It will be useful for our subsequent discussion to define blocksizes(X, n) to be a vector whose pth element is size(getblock(X, p, 1), n)."
},

{
    "location": "block_arrays_ext.html#Why-bother?-1",
    "page": "BlockArrays extensions",
    "title": "Why bother?",
    "category": "section",
    "text": "The last property is particularly important for our purposes: it will allow us to guarantee that the Cholesky factorisation of a positive definite BlockSymmetric matrix can be represented as block matrix for which blocksizes is exactly the same."
},

{
    "location": "block_arrays_ext.html#Which-block-matrices-can-be-block-symmetric?-1",
    "page": "BlockArrays extensions",
    "title": "Which block matrices can be block symmetric?",
    "category": "section",
    "text": "It is important to ask which block matrices can potentially to be block-symmetric. A necessary condition for a block matrix to be block symmetric is then blocksizes(X, 1) == blocksizes(X, 2). From an implementation perspective this condition is sufficient: BlockSymmetric(X) simply checks that this condition holds, and then enforces that X be block symmetric. This is analogous to the way which Symmetric(Y), for some AbstractMatrix Y, checks that Y is square and then enforces Y[p, q] == Y[q, p]."
},

{
    "location": "block_arrays_ext.html#Types-under-consideration-1",
    "page": "BlockArrays extensions",
    "title": "Types under consideration",
    "category": "section",
    "text": "The basic types under consideration are:const ABV{T} = AbstractBlockVector{T}\nconst ABM{T} = AbstractBlockMatrix{T}\nconst ABMV{T} = Union{ABV{T}, ABM{T}}\nconst BS{T} = BlockSymmetric{T, <:AbstractBlockMatrix{T}}Furthermore, we require the following wrappers around BlockSymmetric matrices:UpperTriangular{T, <:BS{T}}\nLowerTriangular{T, <:BS{T}}\nLazyPDMat{T, <:BS{T}}For all of the above, we require \"sensible\" implementations of the following unary functions:getblock\nnblocks\nblocksizes\nsetblock!\ncopy\ntranspose (0.6-specific, changes in 0.7 / 1.0)\nctranspose (0.6-specific, changes in 0.7 / 1.0)"
},

]}
