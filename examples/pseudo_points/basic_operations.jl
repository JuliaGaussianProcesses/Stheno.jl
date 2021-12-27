#
# This file just illustrates the nuts-and-bolts of the approximate inference API provided
# by Stheno. It's worth mentally contrasting this with what you would have written to
# perform exact inference (hint: it's remarkably similar).
#

# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# The time-to-first-plot issue means that this might take a little while.
using AbstractGPs, BenchmarkTools, LinearAlgebra, Plots, Random, Stheno
rng = MersenneTwister(123456);

# Construct a Matern-5/2 kernel with lengthscale 0.5 and variance 1.2.
k = 1.2 * transform(Matern52Kernel(), 2.0);

# Construct a zero-mean GP with a simple kernel. Don't worry about the GPC object.
f = Stheno.wrap(GP(k), GPC());

# Specify some locations at which to consider the GP.
N = 5_000
x = vcat(rand(rng, div(N, 2)) * 3, rand(rng, div(N, 2)) * 4 .+ 6)

# Specify the variance of the noise under which we'll make observations of the GP.
# We could also have made this a `Real` to specify isotropic noise.
Σ = Diagonal(rand(rng, N) .+ 0.1)

# Construct marginal distribution over `f` at `x` added to some independent zero-mean
# Gaussian noise with covariance matrix `Σ`.
fx = f(x, Σ)

# Generate a sample from the prior.
y = rand(rng, fx)

# Now pretend that we can't perform inference because `N` is too large.

# Specify some pseudo-point locations.
z = range(-3.0, 13.0; length=150)

# Specify inducing-points. Add a small amount of jitter for numerical stability.
u = f(z, 1e-6)

# Compute the Evidence Lower BOund.
println("The elbo is reasonable tight.")
@show elbo(fx, y, u), logpdf(fx, y)

println("Benchmark logpdf")
display(@benchmark logpdf($fx, $y))
println()

println("Benchmark elbo")
display(@benchmark elbo($fx, $y, $u))
println()

# Compute the approximate posterior process.
f_post = posterior(VFE(), fx, y, u);

# Specify some points at which to plot the approximate posterior.
Npr = 1000
xpr = range(-3.0, 13.0; length=Npr)

# It's possible to work efficently with the approximate posterior marginals.
# Unfortunately it's not possible to work efficiently with the entire posterior process --
# this is limitation of pseudo-point approximations generally, as opposed to a limitation
# of Stheno.
posterior_marginals = marginals(f_post(xpr))

# Visualise the posterior. At the time of writing, it remains important to do this manually
# for the sake of efficiency. If you would like to have a high-level interface similar to
# the one available in the exact inference setting, please feel free to raise an issue, or
# implement it yourself and open a PR!

# Specify our plotting backend.
gr();

# Construct a new plot object.
posterior_plot = plot(legend=nothing);

# Generate several samples from the posterior predictive distribution.
plot!(posterior_plot, f_post(xpr); color=:blue);

# Plot the observations.
scatter!(posterior_plot, x, y;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=1,
    markeralpha=0.2,
);

# Show the plot.
display(posterior_plot);
