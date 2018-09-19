using Stheno, Random
using Stheno: @model



######################## Example 1: Vanilla Toeplitz ########################

@model function vanilla()
    f = GP(EQ())
    y = f + GP(Noise(0.1))
    return f, y
end
f, y = vanilla();

# Create regularly spaced data. Dispatch handles literally everything else.
x = range(-5.0, stop=5.0, length=10000);

# Notice that we don't struggle to construct large covariances. All of these matrices
# are either Toeplitz or SymmetricToeplitz. This structure is exploited to accelerate
# all computations. These covariance matrices are constucted in O(N)-time.
Σff = cov(f(x));
Σfy = xcov(f(x), y(x));
Σyy = cov(y(x));

# These operations are also quite a lot faster. They occur in O(N^2)-time.
rng = MersenneTwister(123456);
ŷ = rand(rng, y(x));
@show logpdf(y(x), ŷ);



######################## Example 2: Block Vanilla Toeplitz ########################

# Suppose that we have two separate blocks of regularly spaced data eg. missing data.
xl, xr = range(-5.0, stop=0.0, length=5000), range(0.0, stop=5.0, length=5000);

# If we inspect the underlying blocks of the covariance, we see they are all still Toeplitz.
using Stheno: unbox
f_xl_xr = BlockGP([f(xl), f(xr)]);
y_xl_xr = BlockGP([y(xl), y(xr)]);
Σff = cov(f_xl_xr);
display(typeof.(unbox(unbox(Σff)).blocks));

# THIS CURRENTLY SEEMS A BIT SLOW! COMPARE TO DENSE PROPERLY!
rng = MersenneTwister(123456);
ŷ = rand(rng, y_xl_xr);
@show logpdf(y_xl_xr, ŷ);



######################### Example 3: Multi-Output Toeplitz ##########################

# Suppose now that we modify our model slightly. We construct a two-output process from
# three latent processes.
@model function simple_multi_output()
    f₁, f₂, f₃ = GP(EQ()), GP(EQ()), GP(EQ())
    y₁, y₂ = f₁ + f₂, f₂ + f₃
    return y₁, y₂
end
y₁, y₂ = simple_multi_output();

# Observe both at x. The joint covariance matrix is still Toeplitz!
y_1x_2x = BlockGP([y₁(x), y₂(x)]);
Σ12 = cov(y_1x_2x);
display(typeof.(unbox(unbox(Σ12)).blocks));

# Same again for the missing data.
y_1xl_2xr = BlockGP([y₁(xl), y₂(xr)]);
Σ12 = cov(y_1xl_2xr);
display(typeof.(unbox(unbox(Σ12)).blocks));



######################### Example 4: Temporally mis-aligned data ##########################

# What if each output has the same sampling rate, but the data are misaligned?
x₁ = range(-5.0, stop=5.0, length=5000);
x₂ = x₁ .+ 0.3;

# The joint covariance still comprises Toeplitz matrices.
y_x1_x2 = BlockGP([y₁(x₁), y₂(x₂)]);
Σ12 = cov(y_x1_x2);
display(typeof.(unbox(unbox(Σ12)).blocks));



######################### Example 5: Partial structure ###########################

# Suppose that we have some regularly sampled data, and some structure-less data.
x′ = randn(rng, 100);

# Notice that Σ_x_x′ comprised three dense blocks, and one Toeplitz. This is because
# cov(f(x′)) is dense, xcov(f(x), f(x′)) / xcov(f(x′), f(x)) is dense, whilst cov(f(x))
# is Toeplitz.
f_x_x′ = BlockGP([f(x), f(x′)]);
Σ_x_x′ = cov(f_x_x′);
display(typeof.(unbox(unbox(Σ_x_x′)).blocks)) 

# Note that any combination of these things will (should) also yield some combination of
# dense and Toeplitz matrices.
