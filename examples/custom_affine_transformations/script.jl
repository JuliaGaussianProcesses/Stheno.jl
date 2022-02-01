# # Custom Affine Transformations
#
# This page explains how to implement your own affine transformation operations by way of
# example.
# You can add your own custom affine transformations into Stheno using the same
# mechanism as all of the existing transformations (addition, multiplication, composition,
# etc).
# First, load up the relevant packages.

using AbstractGPs
using LinearAlgebra
using Stheno


# ## The Affine Transformation

# Suppose that, for some reason, you wish to implement the affine transformation of a single
# process `f` given by `(Af)(x) = f(x) + f(x + 3) - 2`.
# In order to define this transformation, first create a function which accepts `f` and
# returns a `DerivedGP`:
using Stheno: AbstractGP, DerivedGP, SthenoAbstractGP

A(f::SthenoAbstractGP) = DerivedGP((A, f), f.gpc)

# The first argument to `DerivedGP` contains `A` itself and any data needed to fully
# specify the process results from this transformation. In this case the only piece of
# information required is `f`, but really any data can be put in this argument.
# For example, if we wished to replace the translation of `-3` by a parameter, we could do
# so, and make it a part of this first argument.
#
# The second argument is the book-keeping object that needs to be passed around in order to
# know how to compute covariances properly.
# It is because of this argument that we restrict the accepted GPs to be `SthenoAbstractGP`,
# as we can safely assume that these have a `gpc` field.

# We'll now define a type alias in order to simplify some methods later on:
const A_args = Tuple{typeof(A), SthenoAbstractGP};


# ## Most Important Methods

# We must now define methods of three functions on
# `A_args`: `mean`, `cov`, and `var`.
# First the `mean` -- this method should accept both an `A_args` and an `AbstracVector`, and
# return the mean vector of `A(f)` at `x`.
# Some textbook calculations reveal that this is

Stheno.mean((A, f)::A_args, x::AbstractVector) = mean(f, x) .+ mean(f, x .+ 3) .- 2

# The first argument here is _always_ going to be precisely the tuple of arguments passed
# into the `DerivedGP` constructor above.
# You can assume that you can compute any statistics of `f` that the AbstractGPs API
# provides.

# We now turn our attention to `cov`. The first method we consider is
# `cov(args::A_args, x::AbstractVector, y::AbstractVector)`, which should return the
# cross-covariance matrix between all pairs of points in `x` and `y` under the transformed
# process, `A(f)`.
# Again, some standard manipulations reveal that this covariance is given by

function Stheno.cov((A, f)::A_args, x::AbstractVector, y::AbstractVector)
    return cov(f, x, y) + cov(f, x, y .+ 3) + cov(f, x .+ 3, y) + cov(f, x .+ 3, y .+ 3)
end

# The last substantially new method to implement is
# `cov(args::A_args, g::AbstractGP, x::AbstractVector, y::AbstractVector)`, which should
# return the cross-covariance matrix between `A(f)` at `x` and `g` at `y`.
# When implementing this method, you can assume you have access to functions like
# `cov(f, g, x, y)` etc:

function Stheno.cov((A, f)::A_args, g::AbstractGP, x::AbstractVector, y::AbstractVector)
    return cov(f, g, x, y) + cov(f, g, x .+ 3, y)
end

# ## Additional (Required) Methods

# There are a number of other methods that you should implement.
# These are all just special cases or slight modifications of the three methods above, and
# should be straightforward to implement given that you've implemented the above methods.

# First, lets build a GPPP containing an instance of our transformation so that some
# properties can be verified.
# The definition of the methods being implemented is demonstrated by checking an
# equality after defining each method.

gppp = @gppp let
    f = GP(SEKernel())
    Af = A(f)
end

# Also create some input vectors.

x_f = GPPPInput(:f, randn(3))
y_f = GPPPInput(:f, randn(6))
x_Af = GPPPInput(:Af, randn(3))
y_Af = GPPPInput(:Af, randn(6))
z_Af = GPPPInput(:Af, randn(3))


# The covariance matrix at a single pair of inputs:

function Stheno.cov((A, f)::A_args, x::AbstractVector)
    return cov(f, x) + cov(f, x, x .+ 3) + cov(f, x .+ 3, x) + cov(f, x .+ 3)
end

cov(gppp, x_Af, x_Af) ≈ cov(gppp, x_Af)


# The diagonal of the covariance matrix at a single pair of inputs:

function Stheno.var((A, f)::A_args, x::AbstractVector)
    return var(f, x) + var(f, x .+ 3) + var(f, x, x .+ 3) + var(f, x .+ 3, x)
end

var(gppp, x_Af) ≈ diag(cov(gppp, x_Af))


# The diagonal of the cross-covariance matrix for equal-length inputs:

function Stheno.var((A, f)::A_args, x::AbstractVector, y::AbstractVector)
    return var(f, x, y) + var(f, x, y .+ 3) + var(f, x .+ 3, y) + var(f, x .+ 3, y .+ 3)
end

var(gppp, x_Af, z_Af) ≈ diag(cov(gppp, x_Af, z_Af))


# The diagonal of the cross-covariance between different processes for equal-length inputs:

function Stheno.var((A, f)::A_args, g::AbstractGP, x::AbstractVector, y::AbstractVector)
    return var(f, g, x, y) + var(f, g, x .+ 3, y)
end

var(gppp, x_Af, x_f) ≈ diag(cov(gppp, x_Af, x_f))


# `cov` and `var` between processes when `Af`'s arguments are the second argument, rather
# than the first:

function Stheno.cov(g::AbstractGP, (A, f)::A_args, x::AbstractVector, y::AbstractVector)
    return cov(g, f, x, y) + cov(g, f, x, y .+ 3)
end

cov(gppp, x_f, x_Af) ≈ cov(gppp, x_Af, x_f)'

function Stheno.var(g::AbstractGP, (A, f)::A_args, x::AbstractVector, y::AbstractVector)
    return var(g, f, x, y) + var(g, f, x, y .+ 3)
end

var(gppp, x_f, x_Af) ≈ var(gppp, x_Af, x_f)


# ## Checking Your Implementation

# Given the numerous methods above, it's a really good idea to utilise the functionality
# provided by AbstractGPs.jl to check that you've implemented them all consistently with one
# another.

using AbstractGPs.TestUtils: test_internal_abstractgps_interface
using Random

rng = MersenneTwister(123456);
test_internal_abstractgps_interface(rng, gppp, x_Af, y_Af);
test_internal_abstractgps_interface(rng, gppp, x_Af, y_f);
test_internal_abstractgps_interface(rng, gppp, x_f, y_Af);

# Roughly speaking, provided that you've implemented the first three methods correctly, this
# test ought to catch any glaring problems if you've made a mistake in the rest.
# If course, it won't check that your implementations of the first three methods correctly
# implement the desired affine transformation, so you should write whatever tests you need
# in order to convince yourself of that.
