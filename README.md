# Stheno

[![Build Status](https://github.com/willtebbutt/Stheno.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/Stheno.jl/actions)
[![codecov.io](http://codecov.io/github/JuliaGaussianProcesses/Stheno.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaGaussianProcesses/Stheno.jl?branch=master)
[![](https://img.shields.io/badge/docs-blue.svg)](https://JuliaGaussianProcesses.github.io/Stheno.jl/dev)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Stheno is designed to make doing some kinds of non-standard things with Gaussian processes straightforward.
It has a simple modeling syntax, is inherently able to handle both multi-input and multi-output problems, and trivially supports interdomain pseudo-point approximations.

[We also have a Python version of the package](https://github.com/wesselb/stheno)

Please open issues liberally -- if there's anything that's unclear or doesn't work, we would very much like to know about it.

__Installation__ - `] add Stheno`.

[JuliaCon 2019 Talk](https://www.youtube.com/watch?v=OO3BBkGEMV8)

[Go faster with TemporalGPs.jl](https://github.com/willtebbutt/TemporalGPs.jl/)



## A Couple of Examples

The primary sources of information regarding this package are the [documentation](https://juliagaussianprocesses.github.io/Stheno.jl/stable) and the examples folder, but here are a couple of flashy examples to get started with.

Please raise an issue immediately if either of these examples don't work -- they're not currently included in CI, so there's always a higher chance that they'll be outdated than the internals of the package.

In this first example we define a simple Gaussian process, make observations of different bits of it, and visualise the posterior. We are trivially able to condition on both observations of both `f₁` _and_ `f₃`, which isn't something that's typically straightforward.
```julia

#
# We'll get going by setting up our model, generating some toy observations,
# and constructing the posterior processes produced by conditioning on these
# observations.
#

using AbstractGPs, Stheno, Random, Plots

# Create a pseudo random number generator for reproducibility.
rng = MersenneTwister(123456);

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
# This `GPPP` object is just an `AbstractGPs.AbstractGP` object.
f = @gppp let
    f₁ = GP(randn(rng), SEKernel())
    f₂ = GP(SEKernel())
    f₃ = f₁ + f₂
end;

# Sample `N₁` / `N₂` locations at which to measure `f₁` / `f₃`.
N₁, N₃ = 10, 11;
X₁ = GPPPInput(:f₁, rand(rng, N₁) * 10);
X₃ = GPPPInput(:f₃, rand(rng, N₃) * 10);
X = BlockData(X₁, X₃);

# Pick out the bits of `f` that we're interested in, and the variance
# of the noise under which we plan to measure them.
σ² = 1e-2
fx = f(X, 1e-2);

# Sample toy observations of `f₁` / `f₃` at `X₁` / `X₃`.
y = rand(rng, fx);

# You could work backwards to figure out which elements of `y` correspond to
# which of the elements of `X`, but `Stheno.jl` provides methods of `split` to
# do this for you.
ŷ₁, ŷ₃ = split(X, y);

# Compute the logpdf of the observations. Notice that this looks exactly like
# what you would write in AbstractGPs.jl.
l = logpdf(fx, y)

# Compute the ELBO of the observations, with pseudo-points at the same
# locations as the observations. Could have placed them in any of the processes
# in f, even in f₂.
l ≈ elbo(fx, y, f(X))

# Compute the posterior. This is just an `AbstractGPs.PosteriorGP`.
f′ = posterior(fx, y);



#
# The are various things that we can do with a Stheno model.
#

# Sample jointly from the posterior over all of the processes.
Np, S = 500, 11;
X_ = range(-2.5, stop=12.5, length=Np);
Xp1 = GPPPInput(:f₁, X_);
Xp2 = GPPPInput(:f₂, X_);
Xp3 = GPPPInput(:f₃, X_);
Xp = BlockData(Xp1, Xp2, Xp3);
f′_Xp = rand(rng, f′(Xp, 1e-9), S);

# Chop up posterior samples using `split`.
f₁′Xp, f₂′Xp, f₃′Xp = split(Xp, f′_Xp);



#
# We make use of the plotting recipes in AbstractGPs to plot the marginals,
# and manually plot the joint posterior samples.
#

# Instantiate plot and chose backend.
plotly();
posterior_plot = plot();

# Plot posterior over f1.
plot!(posterior_plot, X_, f′(Xp1); color=:red, label="f1");
plot!(posterior_plot, X_, f₁′Xp; color=:red, label="", alpha=0.2, linewidth=1);
scatter!(posterior_plot, X₁.x, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

# Plot posterior over f2.
plot!(posterior_plot, X_, f′(Xp2); color=:green, label="f2");
plot!(posterior_plot, X_, f₂′Xp; color=:green, label="", alpha=0.2, linewidth=1);

# Plot posterior over f3
plot!(posterior_plot, X_, f′(Xp3); color=:blue, label="f3");
plot!(posterior_plot, X_, f₃′Xp; color=:blue, label="", alpha=0.2, linewidth=1);
scatter!(posterior_plot, X₃.x, ŷ₃;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

display(posterior_plot);

```
![](https://github.com/willtebbutt/stheno_models/blob/master/exact/process_decomposition.png)

In the above figure, we have visualised the posterior distribution of all of the processes. Bold lines are posterior means, and shaded areas are three posterior standard deviations from these means. Thin lines are samples from the posterior processes.

In this next example we make observations of two different noisy versions of the same latent process. Again, this is just about doable in existing GP packages if you know what you're doing, but isn't straightforward.

```julia

using AbstractGPs, Stheno, Random, Plots

# Create a pseudo random number generator for reproducibility.
rng = MersenneTwister(123456);

# Construct a Gaussian Process Probabilistic Programme,
# which is just an AbstractGP.
f = @gppp let

    # Define a smooth latent process that we wish to infer.
    f = GP(SEKernel())

    # Define the two noise processes described.
    g = x->sin.(x) .- 5.0 .+ sqrt.(abs.(x))
    noise1 = sqrt(1e-2) * GP(WhiteKernel()) + g
    noise2 = sqrt(1e-1) * GP(3.5, WhiteKernel())

    # Define the processes that we get to observe.
    y1 = f + noise1
    y2 = f + noise2
end;

# Generate some toy observations of `y1` and `y2`.
X1 = GPPPInput(:y1, rand(rng, 3) * 10);
X2 = GPPPInput(:y2, rand(rng, 10) * 10);
X = BlockData(X1, X2);
y = rand(rng, f(X));
ŷ1, ŷ2 = split(X, y);

# Compute the posterior GPPP.
f′ = posterior(f(X), y);

# Sample jointly from the posterior processes.
X_ = range(-2.5, stop=12.5, length=500);
Xp_f = GPPPInput(:f, X_);
Xp_y1 = GPPPInput(:y1, X_);
Xp_y2 = GPPPInput(:y2, X_);
Xp = BlockData(Xp_f, Xp_y1, Xp_y2);

# Sample jointly from posterior over process, and split up the result.
f′Xp, y1′Xp, y2′Xp = split(Xp, rand(rng, f′(Xp, 1e-9), 11));

# Compute and split up posterior marginals. We're not using the plotting
# recipes from AbstractGPs here, to make it clear how one might compute the
# posterior marginals manually.
ms = marginals(f′(Xp, 1e-9));
μf′, μy1′, μy2′ = split(Xp, mean.(ms));
σf′, σy1′, σy2′ = split(Xp, std.(ms));

# Instantiate plot and chose backend
plotly();
posterior_plot = plot();

# Plot posterior over y1.
plot!(posterior_plot, X_, μy1′; color=:red, ribbon=3σy1′, label="", fillalpha=0.3);
plot!(posterior_plot, X_, y1′Xp; color=:red, label="", alpha=0.2, linewidth=1);
scatter!(posterior_plot, X1.x, ŷ1;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 1",
);

# Plot posterior over y2.
plot!(posterior_plot, X_, μy2′; color=:green, ribbon=3σy2′, label="", fillalpha=0.3);
plot!(posterior_plot, X_, y2′Xp; color=:green, label="", alpha=0.2, linewidth=1);
scatter!(posterior_plot, X2.x, ŷ2;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 2",
);

# Plot posterior over f.
plot!(posterior_plot, X_, μf′; color=:blue, ribbon=3σf′, label="f", fillalpha=0.3);
plot!(posterior_plot, X_, f′Xp; color=:blue, label="", alpha=0.2, linewidth=1);

display(posterior_plot);

```
![](https://github.com/willtebbutt/stheno_models/blob/master/exact/simple_sensor_fusion.png)

As before, we visualise the posterior distribution through its marginal statistics and joint samples. Note that the posterior samples over the unobserved process are (unsurprisingly) smooth, whereas the posterior samples over the noisy processes still look uncorrelated and noise-like.

See the docs for more examples.



## Hyperparameter learning and inference

Fortunately, there is really no need for this package to explicitly provide support for hyperparameter optimisation as the functionality is already available elsewhere -- it's sufficient that it plays nicely with other standard packages in the ecosystem such as [Zygote.jl](https://github.com/FluxML/Zygote.jl/) (reverse-mode algorithmic differentiation), [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) (non-linear optimisation), [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl/) (Hamiltonian Monte Carlo / NUTS), [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl/), [Soss.jl](https://github.com/cscherrer/Soss.jl/) / [Turing.jl](https://github.com/TuringLang/Turing.jl). For concrete examples of the use of each of these packages in conjunction with Stheno, see the `Getting Started` section of the [(dev) docs](https://juliagaussianprocesses.github.io/Stheno.jl/dev).



## Non-Gaussian problems

As with hyperparmeter learning, Stheno doesn't support non-Gaussian likelihoods directly.
Rather, we expect users to obtain this functionality through the general functionality provided in `AbstractGPs`.
This part of the JuliaGaussianProcesses ecosystem is under development in [ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/) -- everything there should interoperate well with Stheno.jl.


## GPs + Deep Learning

Again, explicit support not included. Rather just use Stheno in conjunction with [Flux.jl](https://github.com/FluxML/Flux.jl) and related packages.
