# GP API

This documents the user-facing API as it relates to the `GP` object.

This is a more thorough introduction to the internals than the Getting Started guide, which should be refered to if you are new to Stheno.jl. It's somewhere between a reference document and a tutorial.


## GP

The primitive `GP` type is one of the core components of Stheno.jl. A `GP` should be thought of as a distribution over real-valued functions, in the same way that a `Distributions.Normal` is a distribution over real numbers, and `Distibutions.MvNormal` is a distribution over real-valued vectors.

```@docs
GP
```



## FiniteGP

Once constructed, the correct way to interact with a `GP` is via a `FiniteGP`, which is just the multivariate Normal given by considering the `GP` at only a finite set of inputs.

```julia
f = GP(Matern52(), GPC())
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

Stheno.jl used to maintain its own collection of kernes. Fortunately, the maintainers of
various Gaussian process-related pakages decided to come together and create
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), which is
now the home for all things kernel-related.

## MeanFunctions
These are implicit. Please refer to the `GP` documentation for details.
