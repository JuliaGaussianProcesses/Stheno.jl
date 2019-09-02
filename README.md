# Stheno

[![Build Status](https://travis-ci.org/willtebbutt/Stheno.jl.svg?branch=master)](https://travis-ci.org/willtebbutt/Stheno.jl)[![codecov.io](http://codecov.io/github/willtebbutt/Stheno.jl/coverage.svg?branch=master)](http://codecov.io/github/willtebbutt/Stheno.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://willtebbutt.github.io/Stheno.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://willtebbutt.github.io/Stheno.jl/dev)

Stheno is designed to make doing non-standard things with Gaussian processes straightforward. It has an intuitive modeling syntax, is inherently able to handle both multi-input and multi-output problems, and trivially supports interdomain pseudo-point approximations.

[We also have a Python version of the package](https://github.com/wesselb/stheno)

Please open issues liberally -- if there's anything that's unclear or doesn't work, we would very much like to know about it.

__Installation__ - `] add Stheno`.

Please note that Stheno's internals have changed quite substantially for version 0.3.0. No user facing functionality should have changed though, but please raise issues liberally if it has.

## A Couple of Examples

We have a [model zoo](https://github.com/willtebbutt/stheno_models), but here are a couple of examples to get you started.

In this first example we define a simple Gaussian process, make observations of different bits of it, and visualise the posterior. We are trivially able to condition on both observations of both `f₁` _and_ `f₃`, which is a very non-standard capability.
```julia

#
# We'll get going by setting up our model, generating some toy observations, and
# constructing the posterior processes produced by conditioning on these observations.
#

using Stheno, Random, Plots
using Stheno: @model

# Create a pseudo random number generator for reproducibility.
rng = MersenneTwister(123456);

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
@model function model()
    f₁ = GP(randn(rng), eq())
    f₂ = GP(eq())
    f₃ = f₁ + f₂
    return f₁, f₂, f₃
end
f₁, f₂, f₃ = model();

# Sample `N₁` / `N₂` locations at which to measure `f₁` / `f₃`.
N₁, N₃ = 10, 11;
X₁, X₃ = rand(rng, N₁) * 10, rand(rng, N₃) * 10;

# Sample toy observations of `f₁` / `f₃` at `X₁` / `X₃`.
σ² = 1e-2
ŷ₁, ŷ₃ = rand(rng, [f₁(X₁, σ²), f₃(X₃, σ²)]);

# Compute the posterior processes. `f₁′`, `f₂′`, `f₃′` are just new processes.
(f₁′, f₂′, f₃′) = (f₁, f₂, f₃) | (f₁(X₁, σ²)←ŷ₁, f₃(X₃, σ²)←ŷ₃);



#
# The are various things that we can do with a Stheno model.
#

# Sample jointly from the posterior over each process.
Np, S = 500, 25;
Xp = range(-2.5, stop=12.5, length=Np);
f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp, 1e-9), f₂′(Xp, 1e-9), f₃′(Xp, 1e-9)], S);

# Compute posterior marginals.
ms1 = marginals(f₁′(Xp));
ms2 = marginals(f₂′(Xp));
ms3 = marginals(f₃′(Xp));

# Pull and mean and std of each posterior marginal.
μf₁′, σf₁′ = mean.(ms1), std.(ms1);
μf₂′, σf₂′ = mean.(ms2), std.(ms2);
μf₃′, σf₃′ = mean.(ms3), std.(ms3);

# Compute the logpdf of the observations.
l = logpdf([f₁(X₁, σ²), f₃(X₃, σ²)], [ŷ₁, ŷ₃])

# Compute the ELBO of the observations, with pseudo-points at the same locations as the
# observations. Could have placed them anywhere we fancy, even in f₂.
l ≈ elbo([f₁(X₁, σ²), f₃(X₃, σ²)], [ŷ₁, ŷ₃], [f₁(X₁), f₃(X₃)])



#
# Stheno has some convenience plotting functionality for GPs with 1D inputs:
#

# Instantiate plot and chose backend.
plotly();
posterior_plot = plot();

# Plot posteriors.
plot!(posterior_plot, f₁′(Xp); samples=S, color=:red, label="f1");
plot!(posterior_plot, f₂′(Xp); samples=S, color=:green, label="f2");
plot!(posterior_plot, f₃′(Xp); samples=S, color=:blue, label="f3");

# Plot observations.
scatter!(posterior_plot, X₁, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
scatter!(posterior_plot, X₃, ŷ₃;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");

display(posterior_plot);
```
![](https://github.com/willtebbutt/stheno_models/blob/master/exact/process_decomposition.png)

[Model Zoo Link](https://github.com/willtebbutt/stheno_models/blob/master/exact/process_decomposition.jl)

In the above figure, we have visualised the posterior distribution of all of the processes. Bold lines are posterior means, and shaded areas are three posterior standard deviations from these means. Thin lines are samples from the posterior processes.

In this next example we make observations of two different noisy versions of the same latent process. Again, this is just about doable in existing GP packages if you know what you're doing, but isn't straightforward.

```julia
using Stheno, Random, Plots
using Stheno: @model, eq, Noise

# Create a pseudo random number generator for reproducibility.
rng = MersenneTwister(123456);

@model function model()

    # Define a smooth latent process that we wish to infer.
    f = GP(eq())

    # Define the two noise processes described.
    noise1 = sqrt(1e-2) * GP(Noise()) + (x->sin.(x) .- 5.0 .+ sqrt.(abs.(x)))
    noise2 = sqrt(1e-1) * GP(3.5, Noise())

    # Define the processes that we get to observe.
    y1 = f + noise1
    y2 = f + noise2

    return f, noise1, noise2, y1, y2
end
f, noise₁, noise₂, y₁, y₂ = model();

# Generate some toy observations of `y1` and `y2`.
X₁, X₂ = rand(rng, 3) * 10, rand(rng, 10) * 10;
ŷ₁, ŷ₂ = rand(rng, [y₁(X₁), y₂(X₂)]);

# Compute the posterior processes.
(f′, y₁′, y₂′) = (f, y₁, y₂) | (y₁(X₁)←ŷ₁, y₂(X₂)←ŷ₂);

# Sample jointly from the posterior processes and compute posterior marginals.
Xp = range(-2.5, stop=12.5, length=500);
f′Xp, y₁′Xp, y₂′Xp = rand(rng, [f′(Xp, 1e-9), y₁′(Xp, 1e-9), y₂′(Xp, 1e-9)], 100);

ms1 = marginals(f′(Xp));
ms2 = marginals(y₁′(Xp));
ms3 = marginals(y₂′(Xp));

μf′, σf′ = mean.(ms1), std.(ms1);
μy₁′, σy₁′ = mean.(ms2), std.(ms2);
μy₂′, σy₂′ = mean.(ms3), std.(ms3);

# Instantiate plot and chose backend
plotly();
posterior_plot = plot();

# Plot posteriors
plot!(posterior_plot, y₁′(Xp); samples=S, sample_seriestype=:scatter, color=:red, label="");
plot!(posterior_plot, y₂′(Xp); samples=S, sample_seriestype=:scatter, color=:green, label="");
plot!(posterior_plot, f′(Xp); samples=S, color=:blue, label="Latent Function");

# Plot observations
scatter!(posterior_plot, X₁, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 1");
scatter!(posterior_plot, X₂, ŷ₂;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 2");

display(posterior_plot);
```
![](https://github.com/willtebbutt/stheno_models/blob/master/exact/simple_sensor_fusion.png)

[Model Zoo Link](https://github.com/willtebbutt/stheno_models/blob/master/exact/simple_sensor_fusion.jl)

As before, we visualise the posterior distribution through its marginal statistics and joint samples. Note that the posterior samples over the unobserved process are (unsurprisingly) smooth, whereas the posterior samples over the noisy processes still look uncorrelated and noise-like.


## Non-Gaussian problems

Stheno is designed for jointly Gaussian problems, and there are no plans to support non-Gaussian likelihoods in the core package. The official stance (if you can call it that) is that since Stheno is trivially compatible with [Turing.jl](https://github.com/TuringLang/), and one should simply embed a Stheno model within a Turing model to solve non-Gaussian problems.

This is not to say that there would be no value in the creation of a separate package that extends Stheno to handle, for example, non-Gaussian likelihoods.

## GPs + Deep Learning

The plan is again not to support the combination of GPs and Deep Learning explicitly, but rather to ensure that Stheno and [Flux.jl](https://github.com/FluxML/Flux.jl) play nicely with one another. Both packages now work with [Zygote.jl](https://github.com/FluxML/Zygote.jl), so you can use that to sort out gradient information.

## Things that are definitely up for grabs
Obviously, improvements to code documentation are always welcome, and if you want to write some more unit / integration tests, please feel free. In terms of larger items that require some attention, here are some thoughts:
- An implementation of SVI from [Gaussian Processes for Big Data](https://arxiv.org/abs/1309.6835).
- Kronecker-factored matrices: this is quite a general issue which might be best be addressed by the creation of a separate package. It would be very helpful to have an implementation of the `AbstractMatrix` interface which implements multiplication, inversion, eigenfactorisation etc, which can then be utilised in Stheno.
- Primitives for multi-output GPs: although Stheno does fundamentally have support for multi-output GPs, in the same way that it's helpful to implement so-called "fat" nodes in Automatic Differentiation systems, it may well be helpful to implement specialised multi-output processes in Stheno for performance's sake.
- Some decent benchmarks: development has not focused on performance so far, but it would be extremely helpful to have a wide range of benchmarks so that we can begin to ensure that time is spent optimally. This would involve comparing against [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl), but also some other non-Julia packages.

If you are interested in any of the above, please either open an issue or PR. Better still, if there's something not listed here that you think would be good to see, please open an issue to start a discussion regarding it.
