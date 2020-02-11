# GP API

This documents the user-facing API as it relates to the `GP` object. Everything that is exported should be included here.

This is a more thorough introduction to the internals than the Getting Started guide, which should be refered to if you are new to Stheno.jl. It's somewhere between a reference document and a tutorial.


## GP

The primitive `GP` type is one of the core components of Stheno.jl. A `GP` should be thought of as a distribution over real-valued functions, in the same way that a `Distributions.Normal` is a distribution over real numbers, and `Distibutions.MvNormal` is a distribution over real-valued vectors.

```@docs
GP
```



## FiniteGP

Once constructed, the correct way to interact with a `GP` is via a `FiniteGP`, which is just the multivariate Normal given by considering the `GP` at only a finite set of inputs.

```julia
f = GP(matern52(), GPC())
x = randn(10)
fx = f(x)
```
here `fx`, to be read as "f at x", is a `FiniteGP` to which the following methods apply:
```@docs
mean(::Stheno.FiniteGP)
cov(::Stheno.FiniteGP)
cov(::Stheno.FiniteGP, ::Stheno.FiniteGP)
marginals(::Stheno.FiniteGP)
rand(::Stheno.AbstractRNG, ::Stheno.FiniteGP, N::Int)
logpdf(::Stheno.FiniteGP, ::AbstractVector{<:Real})
elbo(::Stheno.FiniteGP, y::AbstractVector{<:Real}, ::Stheno.FiniteGP)
```


## Kernels

This is an ever-growing list. Implementing another kernel would make an excellent first PR...

```@docs
EQ
PerEQ
Exp
Matern12
Matern32
Matern52
RQ
Cosine
Linear
Poly
GammaExp
Wiener
WienerVelocity
```

## Transformations of Kernels

```@docs
stretch(::Stheno.Kernel, ::Union{Real, AbstractVecOrMat{<:Real}})
*(::Real, ::Stheno.Kernel)
+(::Stheno.Kernel, ::Stheno.Kernel)
*(::Stheno.Kernel, ::Stheno.Kernel)
```

## MeanFunctions
These are implicit. Please refer to the `GP` documentation for details.
