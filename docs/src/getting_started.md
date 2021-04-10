# Getting Started

Here we document how to do some basic stuff, including learning and inference in kernel parameters, with Stheno.jl. To do this, we that makes use of a variety of packages from the Julia ecosystem. In particular, we'll make use of
- [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) to perform Bayesian inference in our model parameters.
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) for point-estimates of our model parameters.
- [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl) to make it easy to work with our model's parameters, and to ensure that it plays nicely with Optim and AdvancedHMC.

This guide assumes that you know roughly what's going on conceptually with GPs. If you're new to Gaussian processes, I cannot recommend [this video lecture](http://videolectures.net/gpip06_mackay_gpb/) highly enough.



## Exact Inference in a GP in 2 Minutes

This is only a slightly more interesting version of the first example on the
README.
It's slightly more interesting in that we give the kernels some learnable
parameters.

```julia
# Import the packages we'll need for this bit of the demo.
using AbstractGPs
using Stheno

# Short length-scale and small variance.
l1 = 0.4
s1 = 0.2

# Long length-scale and larger variance.
l2 = 5.0
s2 = 1.0

# Specify a GaussianProcessProbabilisticProgramme object, which is itself a GP
# built from other GPs.
f = @gppp let
    f1 = s1 * stretch(GP(Matern52Kernel()), 1 / l1)
    f2 = s2 * stretch(GP(SEKernel()), 1 / l2)
    f3 = f1 + f2
end;

# Generate a sample from f3, one of the processes in f, at some random input locations.
# Add some iid observation noise, with zero-mean and variance 0.05.
const x = GPPPInput(:f3, collect(range(-5.0, 5.0; length=100)));
σ²_n = 0.05;
fx = f(x, σ²_n);
const y = rand(fx);

# Compute the log marginal likelihood of this observation, just because we can.
logpdf(fx, y)
```
`fx` should be thought of as "`f` at `x`", and is just as a multivariate Normal distribution, with zero mean and covariance matrix
```julia
cov(f, x) + σ²_n * I
```
As such samples can be drawn from it, and the log probability any particular value under it can be computed, in the same way that you would an `MvNormal` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

We can visualise `x` and `y` with [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
```julia
using Plots
plt = plot();
scatter!(plt, x.x, y; color=:red, label="");
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples.svg)

It's straightforward to compute the posterior over `f`:
```julia
f_posterior = posterior(fx, y);
```
`f_posterior` is another GP, the posterior over `f` given noisy observations `y` at inputs `x`.

Our plotting recipes aren't quite sophisticated enough at the minute to handle GPPPs properly, but plotting still isn't too much work:
```julia
x_plot = range(-7.0, 7.0; length=1000);
xp = GPPPInput(:f3, x_plot);
ms = marginals(f_posterior(xp));
plot!(
    plt, x_plot, mean.(ms);
    ribbon=3std.(ms), label="", color=:blue, fillalpha=0.2, linewidth=2,
);
plot!(
    plt, x_plot, rand(f_posterior(xp), 10);
    alpha=0.3, label="", color=:blue,
);
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior.svg)

So you've built a simple GP probabilistic programme, performed inference in it, and looked at the posterior.
We've only looked at one component of it though -- we could look at others.
Consider `f2`:
```julia
xp2 = GPPPInput(:f2, x_plot);
ms = marginals(f_posterior(xp2));
plot!(
    plt, x_plot, mean.(ms);
    ribbon=3std.(ms), label="", color=:red, fillalpha=0.2, linewidth=2,
);
plot!(
    plt, x_plot, rand(f_posterior(xp2, 1e-9), 10);
    alpha=0.3, label="", color=:red,
);
display(plt);
```



## Fit a GP with NelderMead in 2 Minutes

Stheno.jl is slightly unusual in that it declines to provide a `fit` or `train` function. Why is this? In short, because there's really no need -- the ecosystem now contains everything that is needed to easily do this yourself. By declining to insist on an interface, Stheno.jl is able to interact with a wide array of tools, that you can use in whichever way you please.

Optim requires that you provide an objective function with a single `Vector{<:Real}` parameter for most of its optimisers.
We'll use ParameterHandling.jl to build one of these in a way that doesn't involve manually writing code to convert between a structured, human-readable, representation of our parameters (in a `NamedTuple`) and a `Vector{Float64}`.

First, we'll put the model from before into a function:
```julia
function build_model(θ::NamedTuple)
    return @gppp let
        f1 = θ.s1 * stretch(GP(SEKernel()), 1 / θ.l1)
        f2 = θ.s2 * stretch(GP(SEKernel()), 1 / θ.l2)
        f3 = f1 + f2
    end
end
```

We've assumed that the parameters will be provided as a `NamedTuple`, so let's build one and check that the model can be constructed:
```julia
using ParameterHandling

θ = (
    # Short length-scale and small variance.
    l1 = positive(0.4),
    s1 = positive(0.2),

    # Long length-scale and larger variance.
    l2 = positive(5.0),
    s2 = positive(1.0),

    # Observation noise variance -- we'll be learning this as well.
    s_noise = positive(0.1),
)
```
We've used `ParameterHandling.jl`s `positive` constraint to ensure that all of the parameters remain positive during optimisation.
Note that there's no magic here, and `Optim` knows nothing about `positive`.
Rather, `ParameterHandling` knows how to make sure that `Optim` will optimise the log of the parameters which we want to be positive.

We can make this happen with the following:
```julia
using ParameterHandling
using ParameterHandling: value, flatten

θ_flat_init, unflatten = flatten(θ);

# Concrete types used for clarity only.
unpack(θ_flat::Vector{Float64}) = value(unflatten(θ_flat))
```
Note that `θ_flat_init` is a `Vector{Float64}`, which is usable within `Optim`. `unflatten` takes it, and reconstructs `θ`. Moreover, if we ask for `value(θ)`, we'll get a version of `θ` with all of the `positive` stuff stripped out, and just leave the `NamedTuple`. So `unpack` takes the flat form of the parameters, and converts them into the form that `build_model` is expecting:
```julia
build_model(unpack(θ_flat_init))
```

We can now easily define a function which accepts the flat form of the parameters, and return the negative log marginal likelihood of the parameters:
```julia
# nlml = negative log marginal likelihood (of θ)
function nlml(θ_flat)
    θ = unpack(θ_flat)
    f = build_model(θ)
    return -logpdf(f(x, θ.s_noise), y)
end
```

We can use any gradient-free optimisation technique from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to find the parameters whose negative log marginal likelihood is minimal:
```julia
using Optim
results = Optim.optimize(nlml, θ_flat_init + randn(5), NelderMead())
θ_opt = unpack(results.minimizer);
```
Note that we just added some noise to the initial values to make the optimisation more interesting.

We can now use this to construct the posterior GP and look at the posterior in comparison to the true posterior with the known hyperparameters
```julia
f_opt = build_model(θ_opt);
f_posterior_opt = posterior(f_opt(x, θ_opt.s_noise), y);
ms_opt = marginals(f_posterior_opt(xp));
plot!(
    plt, x_plot, mean.(ms_opt);
    ribbon=3std.(ms_opt), label="", color=:green, fillalpha=0.2, linewidth=2,
);
plot!(
    plt, x_plot, rand(f_posterior_opt(xp, 1e-9), 10);
    alpha=0.3, label="", color=:green,
);
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior_both.svg)

(Of course the exact posterior has not been recovered because the exact hyperparameters cannot be expected to be recovered given a finite amount of data over a finite width window.)


## Fit a GP with BFGS in 2 minutes

The BFGS algorithm is generally the preferred choice when optimising the hyperparameters of fairly simple GPs. It requires access to the gradient of our `nlml` function, which can be straightforwardly obtained via reverse-mode algorithmic differentiation, which is provided by [Zygote.jl](https://github.com/FluxML/Zygote.jl):

```julia
using Zygote: gradient

# This will probably take a while to get going as Zygote needs to compile.
results = Optim.optimize(
    nlml,
    θ->gradient(nlml, θ)[1],
    θ_flat_init + randn(5),
    BFGS(),
    Optim.Options(
        show_trace=true,
    );
    inplace=false,
)
θ_bfgs = unpack(results.minimizer);
```

Once more visualising the results:
```julia
f_bfgs = build_model(θ_bfgs);
f_posterior_bfgs = posterior(f_bfgs(x, θ_bfgs.s_noise), y);
ms_bfgs = marginals(f_posterior_bfgs(xp));
plot!(
    plt, x_plot, mean.(ms_bfgs);
    ribbon=3std.(ms_bfgs), label="", color=:orange, fillalpha=0.2, linewidth=2,
);
plot!(
    plt, x_plot, rand(f_posterior_bfgs(xp, 1e-9), 10);
    alpha=0.3, label="", color=:orange,
);
display(plt);
```
![img](https://willtebbutt.github.io/resources/samples_posterior_bfgs.svg)

Notice that the two optimisers produce (almost) indistinguishable results.


## Inference with NUTS in 2 minutes

[AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl/) provides a state-of-the-art implementation of the No-U-Turns sampler, which we can use to perform approximate Bayesian inference in the hyperparameters of the GP.
This is slightly longer than the previous examples, but it's all set up associated with AdvancedHMC, which is literally a copy-paste from that package's README:
```julia
using AdvancedHMC, Zygote

# Define the log marginal joint density function and its gradient
ℓπ(θ) = -nlml(θ) - 0.5 * sum(abs2, θ)
function ∂ℓπ∂θ(θ)
    lml, back = Zygote.pullback(ℓπ, θ)
    ∂θ = first(back(1.0))
    return lml, ∂θ
end

# Sampling parameter settings
n_samples, n_adapts = 500, 20

# Draw a random starting points
θ0 = randn(5)

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(5)
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
int = Leapfrog(find_good_eps(h, θ0))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ))

# Perform inference.
samples, stats = sample(h, prop, θ0, n_samples, adaptor, n_adapts; progress=true)

# Inspect posterior distribution over hyperparameters.
hypers = unpack.(samples);
h_l1 = histogram(getindex.(hypers, :l1); label="l1");
h_l2 = histogram(getindex.(hypers, :l2); label="l2");
h_s1 = histogram(getindex.(hypers, :s1); label="s1");
h_s2 = histogram(getindex.(hypers, :s2); label="s2");
display(plot(h_l1, h_l2, h_s1, h_s2; layout=(2, 2)));
```
![img](https://willtebbutt.github.io/resources/posterior_hypers.svg)

As expected, the sampler converges to the posterior distribution quickly.
One could combine this code with that from the previous sections to make predictions under the posterior over the hyperparameters.




## Conclusion

So you now know how to fit GPs using Stheno.jl, and to investigate their posterior distributions. It's also straightforward to utilise Stheno.jl inside probabilistic programming frameworks like Soss.jl and Turing.jl (see examples folder).
