using Revise
using Stheno, Plots
plotly();



############################ Modeling and Approximate Inference ############################

# A vanilla noisy regression model.
gpc = GPC();
f = GP(EQ(), gpc);
y = f + GP(Noise(0.1), gpc);

# Make the problem concrete.
rng, N, P, M = MersenneTwister(123456), 100, 300, 25;
X, Xp, Z = linspace(-7.5, 7.5, N), linspace(-10.0, 10.0, P), linspace(-10.0, 10.0, M);
ŷ = rand(rng, y, X);

# Compute exact posterior processes + corresponding marginals.
f′, y′ = (f, y) | (y(X) ← ŷ);
f′μ, f′σ = marginals(f′, Xp);
y′μ, y′σ = marginals(y′, Xp);

# Compute approximate posterior processes + corresponding marginals.
m′u, Σ′uu = Stheno.optimal_q([f], [X], BlockVector([ŷ]), [f], [Z], sqrt(1e-1));
conditioner = Stheno.Titsias(f, Z, m′u, Σ′uu, gpc);
fq = f | conditioner;
fqμ, fqσ = marginals(fq, Xp);

yq = y | conditioner;
yqμ, yqσ = marginals(yq, Xp);



############################ Plotting ############################

# Plot observations.
posterior_plot = plot();
scatter!(posterior_plot, X, ŷ;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Observations");

# Plot exact posterior marginals for noisy process and latent process.
plot!(posterior_plot, Xp, [y′μ y′μ];
    linewidth=0.0,
    fillrange=[y′μ .- 3 .* y′σ, y′μ .+ 3 * y′σ],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
plot!(posterior_plot, Xp, f′μ;
    linecolor=:blue,
    linewidth=2.0,
    label="f′");
plot!(posterior_plot, Xp, [f′μ f′μ];
    linewidth=0.0,
    fillrange=[f′μ .- 3 .* f′σ, f′μ .+ 3 * f′σ],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");

# Plot the approximate posterior marginals over the noisy process.
plot!(posterior_plot, Xp, yqμ;
    linecolor=:cyan,
    linewidth=2.0,
    label="qy");
plot!(posterior_plot, Xp, [yqμ yqμ];
    linewidth=0.0,
    fillrange=[yqμ .- 3 .* yqσ, yqμ .+ 3 * yqσ],
    fillalpha=0.3,
    fillcolor=:cyan,
    label="");

# Plot the approximate posterior marginals over the latent function.
scatter!(posterior_plot, Z, zeros(M);
    markercolor=:black,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z");
plot!(posterior_plot, Xp, fqμ;
    linecolor=:green,
    linewidth=2.0,
    label="qf");
plot!(posterior_plot, Xp, [fqμ fqμ];
    linewidth=0.0,
    fillrange=[fqμ .- 3 .* fqσ, fqμ .+ 3 * fqσ],
    fillalpha=0.3,
    fillcolor=:green,
    label="");

display(posterior_plot);
