# Getting Started

Here we document how to achieve the basic things that any GP package aught to be able to do. We lean heavily on the rest of the Julia ecosystem for each of these examples -- this page really exemplifies the way in which different packages play together nicely in the Julia!

This guide assumes that you know roughly what's going on conceptually with GPs. If you're new to Gaussian processes, I cannot recommend [this video lecture](http://videolectures.net/gpip06_mackay_gpb/) highly enough.

## Exact Inference in a GP in 2 Minutes

While Stheno offers some bells and whistles that other GP frameworks do not, it also offers the same functionality as a usual GP framework.

```julia
using Stheno

# Choose the length-scale and variance of the process.
l = 0.4
σ² = 1.3

# Construct a kernel with this variance and length scale.
k = σ² * stretch(matern52(), 1 / l)

# Specify a zero-mean GP with this kernel. Don't worry about the GPC object.
f = GP(k, GPC())

# Generate a sample from this GP at some random input locations.
# Add some iid observation noise, with zero-mean and variance 0.05.
const x = randn(100)
σ²_n = 0.05
fx = f(x, σ²_n)
const y = rand(fx)

# Compute the log marginal likelihood of this observation, just because we can.
logpdf(fx, y)
```
`fx` should be thought of as "`f` at `x`", and is just as a multivariate Normal distribution, with zero mean and covariance matrix
```julia
Stheno.pairwise(k, x) + σ² * I
```
As such samples can be drawn from it, and the log probability any particular value under it can be computed, in the same way that you would an `MvNormal` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

We can visualise `x` and `y` with [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
```julia
using Plots
plt = plot();
scatter!(plt, x, y; color=:red, label="");
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples.svg)

It's straightforward to compute the posterior over `f`:
```julia
f_posterior = f | Obs(fx, y)
```
`f_posterior` is another GP, the posterior over `f` given noisy observations `y` at inputs `x`. Equivalently:
```julia
f_posterior = f | (fx ← y) # ← is \leftarrow[TAB]
```
This is just syntactic sugar for the above. You can use it, or not, the choice is entirely your own.

[Plots.jl](https://github.com/JuliaPlots/Plots.jl) knows how to plot GPs, so it's straightforward to look at the posterior:
```julia
x_plot = range(-4.0, 4.0; length=1000);
plot!(plt, f_posterior(x_plot); samples=10, label="", color=:blue);
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior.svg)


## Fit a GP with NelderMead in 2 Minutes

Stheno.jl is slightly unusual in that it declines to provide a `fit` or `train` function. Why is this? In short, because it's hard to design a one-size-fits-all interface for training a GP that composes well with the rest of the tools in the Julia ecosystem, and you _really_ want to avoid creating any impediments to interacting with other tools in the ecosystem.

Here we demonstrate the simplest most low-level way to work with Stheno, in which everything is done manually. This example is to demonstrate that the previous section provides all of the basic building blocks that you _need_ to solve regression problems with GPs.

```julia
function unpack(θ)
    σ² = exp(θ[1]) + 1e-6
    l = exp(θ[2]) + 1e-6
    σ²_n = exp(θ[3]) + 1e-6
    return σ², l, σ²_n
end

# nlml = negative log marginal likelihood (of θ)
function nlml(θ)
    σ², l, σ²_n = unpack(θ)
    k = σ² * stretch(matern52(), 1 / l)
    f = GP(k, GPC())
    return -logpdf(f(x, σ²_n), y)
end
```

Hopefully it's clear what we mean by low-level here. We've manually defined a function to unpack a parameter vector `θ` and use this to construct a function that computes the log marginal probability of `y` for any particular `θ`. We can use a gradient-free optimisation technique from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to find the parameters whose log marginal likelihood is minimal:
```julia
using Optim
θ0 = randn(3);
results = Optim.optimize(nlml, θ0, NelderMead())
σ²_ml, l_ml, σ²_n_ml = unpack(results.minimizer);
```

We can now use this to construct the posterior GP and look at the posterior in comparison to the true posterior with the known hyperparameters
```julia
k = σ²_ml * stretch(matern52(), 1 / l_ml);
f = GP(k, GPC());
f_posterior_ml = f | Obs(f(x, σ²_n_ml), y);
plot!(plt, f_posterior_ml(x_plot); samples=10, color=:green, label="");
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior_both.svg)

(Of course the exact posterior has not been recovered because the exact hyperparameters cannot be expected to be recovered.)


## Fit a GP with BFGS in 2 minutes

The BFGS algorithm is generally the preferred choice when optimising the hyperparameters of fairly simple GPs. It requires access to the gradient of our `nlml` function, which can be straightforwardly obtained via reverse-mode algorithmic differentiation, which is provided by [Zygote.jl](https://github.com/FluxML/Zygote.jl):

```julia
using Zygote: gradient
θ0 = randn(3);
results = Optim.optimize(nlml, θ->gradient(nlml, θ)[1], θ0, BFGS(); inplace=false)
σ²_bfgs, l_bfgs, σ²_n_bfgs = unpack(results.minimizer);
```

Once more visualising the results:
```julia
k = σ²_bfgs * stretch(matern52(), 1 / l_bfgs);
f = GP(k, GPC());
f_posterior_bfgs = f | Obs(f(x, σ²_n_bfgs), y);
plot!(plt, f_posterior_bfgs(x_plot); samples=10, color=:purple, label="");
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior_bfgs.svg)

Notice that the two optimisers produce (almost) indistinguishable results.


## Inference with NUTS in 2 minutes

[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl/) provides a state-of-the-art implementation of the No-U-Turns sampler, which we can use to perform approximate Bayesian inference in the hyperparameters of the GP. This is slightly longer than the previous examples, but it's all set up associated with AdvancedHMC, which is literally a copy-paste from that package's README:
```julia
using AdvancedHMC, Zygote

# Define the log marginal likelihood function and its gradient
ℓπ(θ) = -nlml(θ)
function ∂ℓπ∂θ(θ)
    lml, back = Zygote.pullback(ℓπ, θ)
    ∂θ = first(back(1.0))
    return lml, ∂θ
end

# Sampling parameter settings
n_samples, n_adapts = 100, 20

# Draw a random starting points
θ0 = randn(3)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(3)
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
int = Leapfrog(find_good_eps(h, θ0))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ))

# Perform inference
samples, stats = sample(h, prop, θ0, n_samples, adaptor, n_adapts; progress=true)

# Inspect posterior distribution over hyperparameters.
hypers = unpack.(samples);
plt_hypers = plot();
plot!(plt_hypers, getindex.(hypers, 1); label="variance");
plot!(plt_hypers, getindex.(hypers, 2); label="length scale");
plot!(plt_hypers, getindex.(hypers, 3); label="obs noise variance");
display(plt_hypers);
```
![img](https://willtebbutt.github.io/resources/posterior_hypers.svg)

As expected, the sampler converges to the posterior distribution quickly. One could combine this code with that from the previous sections to make predictions under the posterior over the hyperparameters.

Also note that we didn't specify a prior over the kernel parameters in this example, so essentially used an improper prior. We could have used a proper prior by appropriately modifying `ℓπ`.



## Integration with Soss.jl

In all of the examples seen so far it has been necessary to manually define the function `unpack` to convert between a vector representation of our hyperparameters and one that is useful for computing the nlml. Moreover, it has been necessary to manually define conversions between unconstrained and constrained parametrisations of our hyperparameters. While this is fairly straightforward for a small number of parameters, it's clearly not a scalable solution.

Instead we can make use of functionality defined in [Soss.jl](https://github.com/cscherrer/Soss.jl) to

- specify priors for the model parameters
- automatically derive transformations between a vector of unconstrained real numbers and a form that Soss models know how to handle



### Specifying a Soss model

```
using Soss

M = @model x begin
    σ² ~ LogNormal(0, 1)
    l ~ LogNormal(0, 1)
    σ²_n ~ LogNormal(0, 1)
    f = Stheno.GP(σ² * stretch(matern52(), 1 / l), Stheno.GPC())
    y ~ f(x, σ²_n + 1e-3)
end
```

The above defines the model `M`, with `LogNormal(0, 1)` priors over each of our hyperparameters.

```
const t = xform(M(x=x), (y=y,))
```

`t` is a function that transforms between a vector of model parameters and a `NamedTuple` that `Soss` understands.

```
function ℓπ(θ)
    (θ_, logjac) = Soss.transform_and_logjac(t, θ)
    return logpdf(M(x=x), merge(θ_, (y=y,))) + logjac
end
nlj(θ) = -ℓπ(θ)
```

`ℓπ` computes the log joint of the hyperparameters and data. The examples that follow we repeat each of the previously discussed examples without ever having to implement the `unpack` function.



### Fit with Nelder Mead: Stheno + Soss + Optim

```
θ0 = randn(3);
results = Optim.optimize(nlj, θ0, NelderMead());
θ, _ = Soss.transform_and_logjac(t, results.minimizer);
```



### Fit with BFGS: Stheno + Soss + Zygote + Optim

```
# Hack to make Zygote play nicely with a particular thing in Distributions.
Zygote.@nograd Distributions.insupport

# Point estimate of parameters via BFGS.
θ0 = randn(3);
results = Optim.optimize(nlj, θ->first(gradient(nlj, θ)), θ0, BFGS(); inplace=false)
θ, _ = Soss.transform_and_logjac(t, results.minimizer);
```



### Inference with NUTS: Stheno + Soss + Zygote + AdvancedHMC

```
# Inference with AdvancedHMC.
function ∂ℓπ∂θ(θ)
    lml, back = Zygote.pullback(ℓπ, θ)
    ∂θ = first(back(1.0))
    return lml, ∂θ
end

# Sampling parameter settings
n_samples, n_adapts = 100, 20

# Draw a random starting points
θ0 = randn(3)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(3)
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
int = Leapfrog(find_good_eps(h, θ0))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ))

# Perform inference using NUTS.
samples, stats = sample(h, prop, θ0, n_samples, adaptor, n_adapts; progress=true)

hypers = first.(Soss.transform_and_logjac.(Ref(t), samples));
```
`hypers` could be plotted in exactly the same manner as before.



## Conclusion

That's it! You now know how to do typical GP stuff in Stheno. In particular how to:

- specify a kernel with a particular length-scale and variance
- construct a GP
- sample from a GP, and specify an observation noise
- compute the log marginal likelihood of some observations
- visualise a simple 1D example
- infer kernel parameters in a variety of ways
- utilise Soss.jl to automatically handle parameters and their transformations

This are just the basic features of Stheno, that you could expect to find in any other GP package. We _haven't_ covered any of the fancy features of Stheno yet though.
