using Stheno, PyPlot

rng = MersenneTwister(123456);
x′ = linspace(-3.0, 3.0, 250);
x = randn(25);
N = length(x);
gpc = GPC();
f1, f2 = GP(x->0.0, Periodic(EQ()), gpc), GP(x->0.0, RQ(1.0), gpc);
f_noise = 1e-1 * GP(x->0.0, Noise(), gpc);
f = f1 + f2 + f_noise;

# Condition on some samples from the prior marginals.
fs, f1s = sample(rng, f(x)), sample(rng, f1(x));

figure("smooth posterior samples");
plot(x, fs, "kx");
plot(x′, sample(rng, (f1 + f2)(x′) | (f(x) ← fs), 10), "r");

figure("f1 posterior samples");
plot(x, fs, "kx");
plot(x′, sample(rng, f1(x′) | (f(x) ← fs), 10), "r");

figure("f2 posterior samples");
plot(x, fs, "kx");
plot(x′, sample(rng, f2(x′) | (f(x) ← fs), 10), "r");

using Stheno, PyPlot
rng = MersenneTwister(123456);
x = linspace(-3.0, 3.0, 100);

# Construct the forward model.
gpc = GPC();
f_long = GP(x->0.0, Transform(EQ(), x->0.2x), gpc);
f_wiggle = GP(x->0.0, Transform(EQ(), x->5.0x), gpc);
f_periodic = GP(x->0.0, Periodic(EQ()), gpc)

f_long_wiggly = f_long + f_wiggle
f_long_periodic = f_long + f_periodic

# # Sample a bit from the prior.
# fs = sample(rng, [f_long_wiggly(x), f_long_periodic(x)], 5)
# figure("wiggly and periodic")
# plot(x, fs[1], "b")
# plot(x, fs[2], "g")

# Do posterior inference and see what it looks like.
x_wiggly = [-1.0, 1.0]
x_periodic = [-0.5, 0.5]
fs = sample(rng, [f_long_wiggly(x_wiggly), f_long_periodic(x_periodic)])

f_long′ = f_long | [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]
f_long_wiggly′ = f_long_wiggly | [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]
f_long_periodic′ = f_long_periodic | [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]
f_periodic′ = f_periodic | [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]
f_wiggle′ = f_wiggle | [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]

samples = sample(
    rng,
    [f_long′(x),
    f_long_wiggly′(x),
    f_long_periodic′(x),
    f_periodic′(x),
    f_wiggle′(x)]
)

# Look at posterior distribution.
figure("wiggly and periodic posteriors")
plot(x_wiggly, fs[1], "bx")
plot(x_periodic, fs[2], "gx")
plot(x, samples[1], "r")
plot(x, samples[2], "b")
plot(x, samples[3], "g")
plot(x, samples[4] .+ samples[1] .- samples[3], "k")
plot(x, samples[5] .+ samples[1] .- samples[2], "k--")
