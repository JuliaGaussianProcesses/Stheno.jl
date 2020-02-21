#
# This file just shows how some of the basic manipulations you can do in Stheno work in
# practice. See the main documentation and the rest of this examples folder for more info.
#

# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# The time-to-first-plot issue means that this might take a little while.
using LinearAlgebra, Plots, Random, Stheno
rng = MersenneTwister(123456)

# Construct a Matern-5/2 kernel with lengthscale 0.5 and variance 1.2.
k = kernel(Matern52(); l=0.5, s=1.2)

# Construct a zero-mean GP with a simple kernel. Don't worry about the GPC object.
f = GP(k, GPC())

# Specify some locations at which to consider the GP.
N = 50
x = rand(rng, N) * 10

# Specify the variance of the noise under which we'll make observations of the GP.
Σ = Diagonal(rand(rng, N) .+ 0.1)

# Construct marginal distribution over `f` at `x` added to some independent zero-mean
# Gaussian noise with covariance matrix `Σ`.
fx = f(x, Σ)

# Generate a sample from the prior.
y = rand(rng, fx)

# Compute the log marginal probability of the sample under the prior.
logpdf(fx, y)

# Do inference: compute the posterior distribution over `f` given we observe it + noise to
# be `y` at locations `x`.
f_post = f | Obs(fx, y)

# Specify some points at which to plot the posterior.
Npr = 1000
xpr = range(-3.0, 13.0; length=Npr)

# Construct the posterior predictive distribution at `xpr`. Add some jitter.
fx_post = f_post(xpr, 1e-9)

# Draw samples from the posterior.
y_post = rand(rng, fx_post)

# Compute the marginal posterior predictive probability of the samples.
logpdf(fx_post, y_post)

# Compute the posterior marginal distributions. (We could equally have done this with `fx`).
post_marginals = marginals(fx_post)
