# # Getting Started
#
# Here we document how to do some basic stuff, including learning and inference in kernel parameters, with Stheno.jl. To do this, we that makes use of a variety of packages from the Julia ecosystem. In particular, we'll make use of
# - [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) to perform Bayesian inference in our model parameters,
# - [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) for point-estimates of our model parameters,
# - [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl) to make it easy to work with our model's parameters, and to ensure that it plays nicely with Optim and AdvancedHMC,
# - [Zygote.jl](https://github.com/FluxML/Zygote.jl/) to compute gradients.

# This guide assumes that you know roughly what's going on conceptually with GPs. If you're
# new to Gaussian processes, an excellent introduction can be found in either
# [this video lecture](http://videolectures.net/gpip06_mackay_gpb/) or
# [this one](https://www.youtube.com/watch?v=92-98SYOdlY).


# ## Exact Inference in a GP in 2 Minutes
#
# This is only a slightly more interesting version of the first example on the README.
# It's slightly more interesting in that we give the kernels some learnable parameters.

# Import the packages we'll need for this bit of the demo.
using AbstractGPs
using LinearAlgebra
using Stheno
using Plots

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
# Add some iid observation noise, with zero-mean and variance 0.02.
const x = GPPPInput(:f3, collect(range(-5.0, 5.0; length=100)));
σ²_n = 0.02;
fx = f(x, σ²_n);
const y = rand(fx);

# Compute the log marginal likelihood of this observation, just because we can.
logpdf(fx, y)

# `fx` should be thought of as "`f` at `x`", and is just as a multivariate Normal
# distribution, with zero mean and covariance matrix
heatmap(cov(f, x) + σ²_n * I)

# As such samples can be drawn from it, and the log probability any particular value under
# it can be computed, in the same way that you would an `MvNormal` from
# [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

# We can visualise `x` and `y` with [Plots.jl](https://github.com/JuliaPlots/Plots.jl)

plt = plot();
scatter!(plt, x.x, y; color=:red, label="");
display(plt)


# It's straightforward to compute the posterior over `f`:
f_posterior = posterior(fx, y);
# `f_posterior` is another GP, the posterior over `f` given noisy observations `y` at inputs `x`.

# The plotting recipes from AbstractGPs.jl can be utilised to easily print whichever
# component of the GPPP is needed.
x_plot = range(-7.0, 7.0; length=1000);
xp = GPPPInput(:f3, x_plot);
plot!(
    plt, x_plot, f_posterior(xp);
    ribbon_scale=3, label="", color=:blue, fillalpha=0.2, linewidth=2,
)
plot!(
    plt, x_plot, rand(f_posterior(xp, 1e-9), 10);
    samples=10, markersize=1, alpha=0.3, label="", color=:blue,
);
plt

# So you've built a simple GP probabilistic programme, performed inference in it, and
# looked at the posterior.
# We've only looked at one component of it though -- we could look at others.
# Consider `f2`:
xp2 = GPPPInput(:f2, x_plot);
plot!(
    plt, x_plot, f_posterior(xp2);
    ribbon_scale=3, label="", color=:red, fillalpha=0.2, linewidth=2,
);
plot!(
    plt, x_plot, rand(f_posterior(xp2, 1e-9), 10);
    alpha=0.3, label="", color=:red,
);
plt




# ## Fit a GP with NelderMead in 2 Minutes

# Stheno.jl is slightly unusual in that it declines to provide a `fit` or `train` function. Why is this? In short, because there's really no need -- the ecosystem now contains everything that is needed to easily do this yourself. By declining to insist on an interface, Stheno.jl is able to interact with a wide array of tools, that you can use in whichever way you please.

# Optim requires that you provide an objective function with a single `Vector{<:Real}` parameter for most of its optimisers.
# We'll use ParameterHandling.jl to build one of these in a way that doesn't involve manually writing code to convert between a structured, human-readable, representation of our parameters (in a `NamedTuple`) and a `Vector{Float64}`.

# First, we'll put the model from before into a function:
function build_model(θ::NamedTuple)
    return @gppp let
        f1 = θ.s1 * stretch(GP(SEKernel()), 1 / θ.l1)
        f2 = θ.s2 * stretch(GP(SEKernel()), 1 / θ.l2)
        f3 = f1 + f2
    end
end


# We've assumed that the parameters will be provided as a `NamedTuple`, so let's build one and check that the model can be constructed:
using ParameterHandling

θ = (
    ## Short length-scale and small variance.
    l1 = positive(0.4),
    s1 = positive(0.2),

    ## Long length-scale and larger variance.
    l2 = positive(5.0),
    s2 = positive(1.0),

    ## Observation noise variance -- we'll be learning this as well.
    s_noise = positive(0.1),
)

# We've used `ParameterHandling.jl`s `positive` constraint to ensure that all of the
# parameters remain positive during optimisation.
# Note that there's no magic here, and `Optim` knows nothing about `positive`.
# Rather, `ParameterHandling` knows how to make sure that `Optim` will optimise the log of
# the parameters which we want to be positive.

# We can make this happen with the following:
using ParameterHandling
using ParameterHandling: value, flatten

θ_flat_init, unflatten = flatten(θ);

# Concrete types used for clarity only.
unpack = value ∘ unflatten;

# We can now easily define a function which accepts the flat form of the parameters, and
# return the negative log marginal likelihood (nlml) of the parameters θ:
function nlml(θ::NamedTuple)
    f = build_model(θ)
    return -logpdf(f(x, θ.s_noise + 1e-6), y)
end


# We can use any gradient-free optimisation technique from
# [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to find the parameters whose
# negative log marginal likelihood is locally minimal:
using Optim
results = Optim.optimize(
    nlml ∘ unpack,
    θ_flat_init + randn(length(θ_flat_init)),
    NelderMead(),
)
θ_opt = unpack(results.minimizer);
# Note that we just added some noise to the initial values to make the optimisation more
# interesting.

# We can now use this to construct the posterior GP and look at the posterior in comparison
# to the true posterior with the known hyperparameters
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
plt

# Of course, the exact posterior has not been recovered because the exact hyperparameters
# cannot be expected to be recovered given a finite amount of data over a finite width
# window.


# ## Fit a GP with BFGS in 2 minutes

# The BFGS algorithm is generally the preferred choice when optimising the hyperparameters
# of fairly simple GPs. It requires access to the gradient of our `nlml` function, which
# can be straightforwardly obtained via reverse-mode algorithmic differentiation, which is
# provided by [Zygote.jl](https://github.com/FluxML/Zygote.jl):

using Zygote: gradient

# This will probably take a while to get going as Zygote needs to compile.
results = Optim.optimize(
    nlml ∘ unpack,
    θ->gradient(nlml ∘ unpack, θ)[1],
    θ_flat_init + 0.1 * randn(length(θ_flat_init)),
    BFGS(),
    Optim.Options(
        show_trace=true,
    );
    inplace=false,
)
θ_bfgs = unpack(results.minimizer);

# Once more visualising the results:
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
plt

# Notice that the two optimisers produce (almost) indistinguishable results.


# ## Inference with NUTS in 2 minutes

# [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl/) provides a
# state-of-the-art implementation of the No-U-Turns sampler, which we can use to perform
# approximate Bayesian inference in the hyperparameters of the GP.
# This is slightly longer than the previous examples, but it's all set up associated with
# AdvancedHMC, which is literally a copy-paste from that package's README:
using AdvancedHMC, Zygote

# Define the log marginal joint density function and its gradient
ℓπ(θ_flat) = -nlml(unpack(θ_flat)) - 0.5 * sum(abs2, θ_flat)
function ∂ℓπ∂θ(θ_flat)
    lml, back = Zygote.pullback(ℓπ, θ_flat)
    ∂θ_flat = first(back(1.0))
    return lml, ∂θ_flat
end

# Sampling parameter settings
n_samples, n_adapts = 500, 20

# Perturb the initialisation a little bit.
θ0_flat = θ_flat_init + 0.1 * randn(length(θ_flat_init))

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(5)
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
int = Leapfrog(find_good_eps(h, θ0_flat))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ))

# Perform inference.
samples, stats = sample(h, prop, θ0_flat, n_samples, adaptor, n_adapts; progress=true)

# Inspect posterior distribution over hyperparameters.
hypers = map(unpack, samples);
h_l1 = histogram(getindex.(hypers, :l1); label="l1");
h_l2 = histogram(getindex.(hypers, :l2); label="l2");
h_s1 = histogram(getindex.(hypers, :s1); label="s1");
h_s2 = histogram(getindex.(hypers, :s2); label="s2");
plot(h_l1, h_l2, h_s1, h_s2; layout=(2, 2))

# As expected, the sampler converges to the posterior distribution quickly.
# One could combine this code with that from the previous sections to make predictions under
# the posterior over the hyperparameters.




# ## Conclusion

# So you now know how to fit GPs using Stheno.jl, and to investigate their posterior
# distributions. It's also straightforward to utilise Stheno.jl inside probabilistic
# programming frameworks like Soss.jl and Turing.jl (see examples folder).
