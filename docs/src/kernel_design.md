# Kernel Design

Stheno.jl provides a compositional approach to providing all of the usual niceties for kernel construction. We outline the basic kernels, how you can choose their length scales / variances, and the ways in which you can combine them.

The interface to use the kernels to compute things isn't strictly user-facing, so please refer to the `Internals` section on kernels for more info about doing things with a kernel once you've built it.

## Base Kernels

Stheno maintains a diverse collection of so-called "base" kernels. These include, but are not limited to, the

- exponentiated quadratic a.k.a. squared exponential, radial basis function (RBF)
- various matern kernels
- rational quadratic

All of these base kernels have unit length scale and variance by default. We'll see shortly how to move beyond this.

The collection of base kernels can be found in `src/GP/kernels.jl` -- at any given point in time it should be considered the definitive source of knowledge regarding the available base kernels. Their implementations are simple, so it's highly recommended to take a look to see what's available.


## Length scales and variances

As discussed, the base kernels don't come equipped with any notion of a length scale of variance. It's simple to provide any of them with such properties though. The approach we take here has the distinct benefits of immediately applying to any new kernels that are implemented, so if you decide to implement a new kernel there's no need to worry about its length scale or variance, just implement it assume they're both `1` and you'll be grand. This is great, as you get ARD and factor-analyis kernels for free when you implement a new kernel!

### Kernel Variance / Amplitude

Given a kernel `k`, and a desired variance `s`, we have that
```julia
(s * k)(x, y) = s * k(x, y)
```
In other words, `s * k` returns a new kernel that's the original kernel with a scaled variance.

### Length Scale

Given a kernel `k`, and a desired inverse length scale `a`, we have that
```julia
stretch(k, a)(x, y) = k(a * x, a * y)
```
which, if you do the maths, you'll see is exactly what you needed to do to implement the length scale of a kernel.

This extends to ARD kernels straightforwardly -- just make `a` an `AbstractVector{<:Real}`. Furthermore it extends to matrices, enabling learning of a low-dimensional representation of the data.


## Kernel Composition

The core piece of functionality that makes Stheno.jl's kernels work is composition. For example, the `Sum` kernel lets you write
```julia
k3 = k1 + k2
```
and the `Product` kernel enables
```julia
k3 = k1 * k2
```

In fact, the variance and `stretch` functionality discussed above is implemented in terms of composite kernels. Take a look towards the bottom of the `src/GP/kernels.jl` file if you're interested how things. The `Sum` and `Product` kernels are good places to start here.
