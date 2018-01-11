using Stheno

rng = MersenneTwister(123456);
x = randn(25);
N = length(x);
f = GP(x->0.0, EQ(), GPC());
fs = sample(rng, f(x));
println(lpdf(f(x), fs))
println(lpdf([f(x)←fs, f(x)←fs]))

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

# Construct the forward model.
gpc = GPC();
f_long = 5.0 * GP(x->0.0, Transform(RQ(1.0), x->0.5x), gpc);
f_wiggle = GP(x->0.0, Transform(Wiener(), x->5.0x), gpc);
f_periodic = GP(x->0.0, Periodic(EQ()), gpc)

f_long_wiggly = f_long + f_wiggle
f_long_periodic = f_periodic + -1.0 * f_long

# Do posterior inference and see what it looks like.
x_wiggly = collect(linspace(0.0, 3.0, 10))
x_periodic = collect(linspace(0.0, 3.0, 5)) .+ 0.5
fs = sample(rng, [f_long_wiggly(x_wiggly), f_long_periodic(x_periodic)])

assignments = [f_long_wiggly(x_wiggly) ← fs[1], f_long_periodic(x_periodic) ← fs[2]]
x = linspace(0.0, 6.0, 1000);
f_long′ = f_long(x) | assignments
f_long_wiggly′ = f_long_wiggly(x) | assignments
f_long_periodic′ = f_long_periodic(x) | assignments
f_periodic′ = f_periodic(x) | assignments
f_wiggle′ = f_wiggle(x) | assignments

μ_f′, σ_f′ = mean_vector(f′), sqrt.(diag(full(cov(f′))))

μ_f_long′, σ_f_long′ = mean_vector(f_long′), sqrt.(diag(full(cov(f_long′))))
μ_f_long_wigg′, σ_f_long_wigg′ = mean_vector(f_long_wiggly′), sqrt.(diag(full(cov(f_long_wiggly′))))
μ_f_wigg′, σ_f_wigg′ = mean_vector(f_wiggle′), sqrt.(diag(full(cov(f_wiggle′))))
μ_f_long_per′, σ_f_long_per′ = mean_vector(f_long_periodic′), sqrt.(diag(full(cov(f_long_periodic′))))
μ_f_per′, σ_f_per′ = mean_vector(f_periodic′), sqrt.(diag(full(cov(f_periodic′))))

plot(x, μ_f_long′, "r", label="long")
plot(x, μ_f_long′ .+ 2 .* σ_f_long′, "r--")
plot(x, μ_f_long′ .- 2 .* σ_f_long′, "r--")

plot(x, μ_f_long_wigg′, "b", label="long + wiggly")
plot(x, μ_f_long_wigg′ .+ 2 .* σ_f_long_wigg′, "b--")
plot(x, μ_f_long_wigg′ .- 2 .* σ_f_long_wigg′, "b--")

plot(x, μ_f_long_per′, "g", label="periodic - long")
plot(x, μ_f_long_per′ .+ 2 .* σ_f_long_per′, "g--")
plot(x, μ_f_long_per′ .- 2 .* σ_f_long_per′, "g--")

plot(x, μ_f_per′, "k", label="periodic")
plot(x, μ_f_per′ .+ 2 .* σ_f_per′, "k--")
plot(x, μ_f_per′ .- 2 .* σ_f_per′, "k--")

plot(x, μ_f_wigg′, "c", label="wiggly")
plot(x, μ_f_wigg′ .+ 2 .* σ_f_wigg′, "c--")
plot(x, μ_f_wigg′ .- 2 .* σ_f_wigg′, "c--")

plot(x_wiggly, fs[1], "bx")
plot(x_periodic, fs[2], "gx")

legend()

plot(x, sample(rng, f_long′, 5), "r")

# Probabilistic model for lpdfs.
μ, σ² = 1.0, 0.5
g(x) = -0.5 * log(2π * σ²) - 0.5 * (x - μ)^2 / σ²
ϕ(x) = (x^2, 1.0)

x, x′ = collect(linspace(-3.0, 3.0, 100)), [0.0, 1.0]
y, y′ = g.(x), g.(x′)

f = GP(x->0.0, Linear(0.0), GPC())
f′ = f(ϕ.(x)) | (f(ϕ.(x′)) ← y′)

μ_f′, σ_f′ = mean_vector(f′), sqrt.(diag(full(cov(f′))))
plot(x, μ_f′, "r")
plot(x, μ_f′ .+ 2 .* σ_f′, "r--")
plot(x, μ_f′ .- 2 .* σ_f′, "r--")
plot(x, g.(x), "g")

