# Interfaces

The primary objects in Stheno are `AbstractGP`s, which represent Gaussian processes. There are two primary concrete subtypes of `AbstractGP`:
- `GP`: an atomic Gaussian process, whose `MeanFunction` and `Kernel` are specified directly.
- `CompositeGP`: a Gaussian process composed of other `AbstractGP`s, whose properties are determined recursively from the `AbstractGP`s of which it is composed.

This documentation provides the information necessary to understand the internals of Stheno, and to extend it with custom functionality.



## AbstractGP

The `AbstractGP` interface enables one to compute quantities required when working with Gaussian processes in practice, namely to compute their `logpdf` and sample from them at particular locations in their domain.

| Function | Brief description |
|:--------------------- |:---------------------- |
| `mean_vector(f, x)` | The mean vector of `f` at inputs `x` |
| `cov(f, x)` | covariance matrix of `f` at inputs `x` |
| `cov(f, x, x′)` | covariance matrix between `f` at `x` and `x′` |
| `cov(f, f′, x, x′)` | cross-covariance matrix between `f` at `x` and `f′` at `x′` |

It should always hold that `cov(f, x) ≈ cov(f, f, x, x)`, but in some critical cases `cov(f, x)` is significantly faster.


`GP` and `CompositeGP` are concrete subtypes of `AbstractGP`, and can be found [here](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/gp.jl) and [here](https://github.com/willtebbutt/Stheno.jl/blob/master/src/composite/composite_gp.jl) respectively.

### diag methods

It is crucial for pseudo-point methods, and for the computation of marginal statistics at a reasonable scale, to be able to compute the diagonal of a given covariance matrix in linear time in the size of its inputs. This, in turn, necessitates that the diagonal of a given cross-covariance matrix can also be computed efficiently as the evaluation of covariance matrices often rely on the evaluation of cross-covariance matrices. As such, we have the following functions:

| Function | Brief description |
|:--------------------- |:---------------------- |
| `cov_diag(f, x)` | `diag(cov(f, x))` |
| `cov_diag(f, x, x′)` | `diag(cov(f, x, x′))` |
| `cov_diag(f, f′, x, x′)` | `diag(cov(f, f′, x, x′))` |

The second and third rows of the table only make sense when `length(x) == length(x′)`, of course.


## GP

We can construct a `GP` in the following manner:

```julia
GP(m, k, gpc)
```
where `m` is its `MeanFunction`, `k` its `Kernel`. `gpc` is a `GPC` object that handles some book-keeping, and is discussed in more depth later (don't worry it's very straightforward, and only mildly annoying).

The `AbstractGP` interface is implemented for `GP`s via operations on their `MeanFunction` and `Kernel`. It is therefore straightforward to extend the range of functionality offered by `Stheno.jl` by simply implementing a new `MeanFunction` or `Kernel` that satisfies their interface, which we detail below.

### MeanFunctions

`MeanFunction`s are unary functions with `Real`-valued outputs with a single-method interface. They must implement `elementwise` (aliased to `ew` for brevity) with the signature
```julia
ew(m::MyMeanFunction, x::AbstractVector)
```
This function applies the `MeanFunction` to each element of `x`, and should return an `AbstractVector{<:Real}` of the same length as `x`. Note that `x` represents a vector of observations, not a single feature vector. Some example implementations can be found [here](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/mean.jl).

Note that while `MeanFunction`s are in principle functions, their interface does not require that we can evaluate `m(x[p])`, only that the "vectorised" `elementwise` function be implemented. This is due to the fact that, in practice, we only ever need the result of `elementwise`.

There are a couple of methods of `GP` which are specialised to particular `MeanFunction`s:
```julia
GP(k::Kernel, gpc::GPC) == GP(ZeroMean(), k, gpc)
GP(c::Real, k::Kernel, gpc::GPC) == GP(ConstMean(c), k, gpc)
```

### Kernels

A `Kernel` is a binary function, returning a `Real`-valued result. `Kernel`s are only slightly more complicated than `MeanFunction`s, having a four-method interface:
```julia
# Binary methods
ew(k::MyKernel, x::AbstractVector, x′::AbstractVector) # "Binary elementwise"
pw(k::MyKernel, x::AbstractVector, x′::AbstractVector) # "Binary pairwise"

# Unary methods
ew(k::MyKernel, x::AbstractVector) # "Unary elementwise"
pw(k::MyKernel, x::AbstractVector) # "Unary pairwise"
```
Again, `ew === elementwise` and `pw === pairwise`.

Note that, as with `MeanFunction`s, the `Kernel` interface does not require that one can actually evaluate `k(x[p], x′[q])`, as in practice this functionality is never _really_ required and would otherwise be extra code to maintain.

We consider each method in turn.

- Binary elementwise: compute `k(x[p], x′[p])` for `p in eachindex(x)`. `x` and `x′` are assumed to be of the same length. Returns a subtype of `AbstractVector{<:Real}`, of the same length as `x` and `x′`.
- Binary pairwise: compute `k(x[p], x′[q])` for `p in eachindex(x)` and `q in eachindex(x′)`. `x` and `x′` need not be of the same length. Returns a subtype of `AbstractMatrix{<:Real}` whose size is `(length(x), length(x′))`.
- Unary elementwise: compute `k(x[p], x[p])` for `p in eachindex(x)`. Returns a subtype of `AbstractVector{<:Real}` of the same length as `x`.
- Unary pairwise: compute `k(x[p], x[q])` for `p in eachindex(x)` and `q in eachindex(x)`. Returns a subtype of `AbstractMatrix{<:Real}` whose size is `(length(x), length(x))`. Crucially, output must be positive definite and (approximately) symmetric.

Example implementations can be found below. Often you'll find that multiple versions of each method are implemented, specialised to different input types. For example, the `EQ` kernel has (at the time of writing) two implementations of each method, one for inputs `AbstractVector{<:Real}`, and one for `ColVecs <: AbstractVector` inputs. These specialisations are for performance purposes.

### Example Kernel implementation

It's straightforward to implement a new kernel yourself: define a new type and implement the two `pw` and `ew` methods required to make it play nicely with everything else in Stheno. This process is broken down below.

```julia
using Stheno
using Stheno: Kernel

struct EQ{Tl<:Real} <: Kernel
    l::Tl
end

_eq(l::Real, xl::Real, xr::Real) = exp(-((xl - xr) / l)^2)
```

The above defines a structure that represents an Exponentiated Quadratic (a.k.a. RBF / Radial Basis Function, Squared Exponential) kernel with length scale `l`. The `_eq` function defines how the kernel operates on a pair of real values given the length-scale and is just a helper function used to define the `pw` and `ew` methods below.

```julia
import Stheno: pw
pw(k::EQ, x::AbstractVector{<:Real}) = _eq.(k.l, x, x')
```

This is one way to implement the unary `pairwise` method for this kernel using Julia's broadcasting functionality. Sampling from the prior and computing the log marginal probability of data is possible given just this method:
```julia
# Construct a GP. See below for info regarding the GPC, but don't worry
# too much about it.
using Stheno: GPC
l = 2.4
f = GP(EQ(l), GPC())

# Sample from the prior and add some iid observation noise with variance 0.1.
x = range(-5.0, 5.0; length=100)
y = rand(f(x, 0.1))

# Compute the log marginal probability of `y`.
logpdf(f(x, 0.1), y)
```

To compute posterior predictive likelihoods, and to sample from the posterior, the binary `pairwise` method is required:
```julia
# Define binary `pairwise` method.
pw(k::EQ, xl::AbstractVector{<:Real}, xr::AbstractVector{<:Real}) = _eq.(k.l, xl, xr')

# Compute posterior process.
f_post = f | (f(x, 0.1) ← y);

# Sample from posterior predictive with a tiny amount of noise for numerical stability.
x_pr = randn(10)
y_pr = rand(f_post(x_pr, 1e-12))

# Compute the log marginal conditional probability of the posterior sample.
logpdf(f_post(x_pr, 1e-12), y_pr)
```

Often the marginal statistics of a GP are helpful, and it is for these (and pseudo-point methods) that the `elementwise` / `ew` methods are required:
```julia
import Stheno: ew
ew(k::EQ, xl::AbstractVector{<:Real}, xr::AbstractVector{<:Real}) = _eq.(k.l, xl, xr)
ew(k::EQ, x::AbstractVector{<:Real}) = ones(length(x))
```
It is now possible to compute the posterior marginal statistics at a large number of points efficiently:
```julia
f_post_marginals = marginals(f_post(range(-10.0, 10.0; length=5_000)))
means = mean.(f_post_marginals)
stds = std.(f_post_marginals)
```
and to use Stheno's plotting functionality for pretty-printing:
```julia
using Plots # Possibly type `]add Plots`
plot(f_post(range(-10.0, 10.0; length=2_000)); samples=10, color=:blue, label="")
scatter!(x, y; markersize=2.0, color=:black, label="")
```

Stheno provides a more general implementation of the Exponentiated Quadratic (EQ) kernel, which is only a minor extension of the above, and can be found in [kernels.jl](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/kernel.jl) alongside various other kernels that are available.

### Why no sensible fallbacks?

Early versions of Stheno required that new kernels also define a method that evaluates the kernel at a pair of inputs (i.e., the functionality provided by the `_eq` function in the example above). This requirement was found to be of minimal use in practice in any situation other than prototyping as, in the vast majority of cases, better performance is obtained by directly implementing `pw` and `ew`. This difference in performance is often sufficiently stark as to render them useless for most practical purposes, meaning that it usually better for Stheno to error and let the user know that an efficient method is missing than to proceed with the fallback.

If you feel this approach is helpful for your work, it is recommended to adapt the code developed in the example above and define a function similar to `_ew` that suits your own needs. Alternatively, if you would _really_ to see this functionality, please raise an issue or open a PR. As with most things in the Julia ecosystem, someone will build it if there is sufficient demand.

### AbstractGP Interface Implementation

Given the above, the `AbstractGP` interface is straightforward to implement for `GPsme`, as each method of `mean_vector` and `cov` can be implemented in terms of `ew` and `pw`. See [here](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/gp.jl) for the implementation.

If you are interested just in working with a single `GP` object, with a known `MeanFunction` and `Kernel`, this is probably as far as you need to go. Simply implement you own fancy `Mean` and `Kernel` objects, or approximations to them, and have some fun / do some research.


## CompositeGP

`CompositeGP`s are constructed as affine transformations of `CompositeGP`s and `GP`s. We describe the implemented transformations below.


### Addition

Given `AbstractGP`s `f` and `g`, we define
```julia
h = f + g
```
to be the `CompositeGP` sastisfying `h(x) = f(x) + g(x)` for all `x`.


### Multiplication

Multiplication of `AbstractGP`s is undefined since the product of two Gaussian random variables is not itself Gaussian. However, we can scale an `AbstractGP` by either a constant or (deterministic) function.
```julia
h = c * f
h = sin * f
```
will both work, and produce the result that `h(x) = c * f(x)` or `h(x) = sin(x) * f(x)`.


### Composition
```julia
h = f ∘ g
```
for some deterministic function `g` is the composition of `f` with `g`. i.e. `h(x) = f(g(x))`.


### conditioning
```julia
h = g | (f(x) ← y)
```
should be read as `h` is the posterior process produced by conditioning the process `g` on having observed `f` at inputs `x` to take values `y`.


### approximate conditioning
TODO (implemented, not documented)

### cross
TODO (implemented, not documented)

## GPC

This book-keeping object doesn't matter from a user's perspective but, unfortunately, we currently expose it to users. Fortunately, it's straightforward to work with. Say you wish to construct a collection of processes:
```julia
# THIS WON'T WORK
f = GP(mf, kf)
g = GP(mg, kg)
h = f + g
# THIS WON'T WORK
```
You should write
```julia
# THIS IS GOOD. PLEASE DO THIS
gpc = GPC()
f = GP(mf, kf, gpc)
g = GP(mg, kg, gpc)
h = f + g
# THIS IS GOOD. PLEASE DO THIS
```
The rule is simple: when constructing `GP` objects that you plan to make interact later in your program, construct them using the same `gpc` object. For example, DON'T do the following:
```julia
# THIS IS BAD. PLEASE DON'T DO THIS
f = GP(mf, kf, GPC())
g = GP(mg, kg, GPC())
h = f + g
# THIS IS BAD. PLEASE DON'T DO THIS
```
The mistake here is to construct a separate `GPC` object for each `GP`. Hopefully, the code errors, but might yield incorrect results.

Alternatively, if you're willing to place your model in a function you can write something like:
```julia
@model function foo(some arguments)
    f1 = GP(mean, kernel)
    f2 = GP(some other mean, some other kernel)
    return f1, f2
end
```
The `@model` macro places a `GPC` on the first line of the function and provides it as an argument to each `GP` constructed. Suggestions for ways to improve/extend this interface are greatly appreciated.
