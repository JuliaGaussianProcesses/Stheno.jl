# Interfaces

The primary objects in Stheno are `AbstractGP`s, which represent Gaussian processes. There are two primary concrete subtypes of `AbstractGP`:
- `GP`: an atomic Gaussian process, whose `MeanFunction` and `Kernel` are specified directly.
- `CompositeGP`: a Gaussian process composed of other `AbstractGP`s, whose properties are determined recursively from the `AbstractGP`s of which it is composed.

This documentation provides the information necessary to understand the internals of Stheno, and to extend it with custom functionality.



## AbstractGP

The `AbstractGP` interface enables one to compute quantities required when working with Gaussian processes in practice, namely to compute their `logpdf` and sample from them at particular locations in their domain.

| Function | Brief description |
|:--------------------- |:---------------------- |
| `mean(f, x)` | The mean vector of `f` at inputs `x` |
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

### Example Kernel implementation

It's straightforward to implement a new kernel yourself: simply following the instructions in [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) and it'll work fine with GPs in Stheno.jl!

### AbstractGP Interface Implementation

Given the above, the `AbstractGP` interface is straightforward to implement for `GPsme`, as each method of `mean` and `cov` can be implemented in terms of `ew` and `pw`. See [here](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/gp.jl) for the implementation.

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
