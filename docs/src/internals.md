# Interfaces

The primary objects in Stheno are some special subtypes of `AbstractGP`. There are three primary concrete subtypes of `AbstractGP`:
- `WrappedGP`: an atomic Gaussian process given by wrapping an `AbstractGP`.
- `CompositeGP`: a Gaussian process composed of other `WrappedGP`s and `CompositeGP`s, whose properties are determined recursively from the GPs of which it is composed.
- `GaussianProcessProbabilisticProgramme` / `GPPP`: a Gaussian process comprising `WrappedGP`s and `CompositeGP`s. This is the primary piece of functionality that users should interact with.

This documentation provides the information necessary to understand the internals of Stheno, and to extend it with custom functionality.



## AbstractGP

[`WrappedGP`](https://github.com/willtebbutt/Stheno.jl/blob/master/src/gp/gp.jl) and [`CompositeGP`](https://github.com/willtebbutt/Stheno.jl/blob/master/src/composite/composite_gp.jl) implement the [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/) API. Please refer to the AbstractGPs.jl docs for more info on this.

### diag methods

It is crucial for pseudo-point methods, and for the computation of marginal statistics at a reasonable scale, to be able to compute the diagonal of a given covariance matrix in linear time in the size of its inputs.
This, in turn, necessitates that the diagonal of a given cross-covariance matrix can also be computed efficiently as the evaluation of covariance matrices often rely on the evaluation of cross-covariance matrices.
As such, we have the following functions in addition to the AbstractGPs API implemented for `WrappedGP`s and `CompositeGP`s:

| Function | Brief description |
|:--------------------- |:---------------------- |
| `var(f, x)` | `diag(cov(f, x))` |
| `var(f, x, x′)` | `diag(cov(f, x, x′))` |
| `var(f, f′, x, x′)` | `diag(cov(f, f′, x, x′))` |

The second and third rows of the table only make sense when `length(x) == length(x′)`, of course.


## WrappedGP

We can construct a `WrappedGP` in the following manner:

```julia
f = wrap(GP(m, k), gpc)

```
where `m` is its `MeanFunction`, `k` its `Kernel`. `gpc` is a `GPC` object that handles some book-keeping, and is discussed in more depth below.

The `AbstractGP` interface is implemented for `WrappedGP`s in terms of the `AbstractGP` that they wrap.
So if you want a particular behaviour, just make sure that the `AbstractGP` that you wrap has the functionality you want.

### Aside: Example Kernel implementation

It's straightforward to implement a new kernel yourself: simply following the instructions in [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) and it'll work fine with GPs in Stheno.jl.

### GPC

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
f = wrap(GP(mf, kf), gpc)
g = wrap(GP(mg, kg), gpc)
h = f + g
# THIS IS GOOD. PLEASE DO THIS
```
The rule is simple: when constructing GPs that you plan to make interact later in your program, construct them using the same `gpc` object. For example, DON'T do the following:
```julia
# THIS IS BAD. PLEASE DON'T DO THIS
f = wrap(GP(mf, kf), GPC())
g = wrap(GP(mg, kg), GPC())
h = f + g
# THIS IS BAD. PLEASE DON'T DO THIS
```
The mistake here is to construct a separate `GPC` object for each `GP`. Hopefully, the code errors, but might yield incorrect results.




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



### cross
```julia
h = cross([f, g])
```
for `WrappedGPs` and `CompositeGP`s `f` and `g`. Think of `cross` as having stacked `f` and `g` together, so that you can work with their joint.

For example, if you wanted to sample jointly from `f` and `g` at locations `x_f` and `x_g`, you could write something like
```julia
fg = cross([f, g])
x_fg = BlockData([x_f, x_g])
rand(fg(x_fg, 1e-6))
```
This particular function isn't part of the user-facing API because it isn't generally needed. It is, however, used extensively in the implementation of the `GaussianProcessProbabilisticProgramme`.



## GPPP

The `GaussianProcessProbabilisticProgramme` is another `AbstractGP` which enables provides a thin layer of convenience functionality on top of `WrappedGP`s and `CompositeGP`s, and is the primary way in which it is intended that users will interact with this package.

A `GPPP` like this
```julia
f = @gppp let
    f1 = GP(SEKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end
```
is equivalent to manually constructing a `GPPP` using `WrappedGP`s and `CompositeGP`s:
```julia
gpc = GPC()
f1 = wrap(GP(SEKernel()), gpc)
f2 = wrap(GP(SEKernel()), gpc)
f3 = f1 + f2
f = Stheno.GPPP((f1=f1, f2=f2, f3=f3), gpc)
```
If you take a look at the `gaussian_process_probabilistic_programming.jl` source, you'll see
that it's essentially just the above, and an implementation of the `AbstactGP`s API on top
of a `GPPP`.

A `GPPP` is essentially just a single GP on an extended input domain:

![](no_luck_catching_them_swans_then.jpeg)
