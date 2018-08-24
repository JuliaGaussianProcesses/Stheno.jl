using Stheno, Random, Plots
using Stheno: @model, Titsias



###########################  Define our model  ###########################

σ² = 1e-1

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
@model function model()

    # Define latent processes.
    f₁ = GP(ConstantMean(randn()), EQ())
    f₂ = GP(EQ())
    f₃ = f₁ + f₂

    # Define noisy versions of latent processes which we are permitted to observe.
    y₁ = f₁ + GP(Noise(σ²))
    y₂ = f₂ + GP(Noise(σ²))
    y₃ = f₃ + GP(Noise(σ²))

    return f₁, f₂, f₃, y₁, y₂, y₃
end

f₁, f₂, f₃, y₁, y₂, y₃ = model();

# Randomly sample `N₁` locations at which to measure `f` using `y1`, and `N2` locations
# at which to measure `f` using `y2`.
rng, N₁, N₃, M = MersenneTwister(123546), 100, 100, 50;
Z = range(-15.0, stop=15.0, length=M);
X₁ = sort(rand(rng, N₁) * 10);
X₃ = sort(rand(rng, N₃) * 10);

# Generate some toy observations of `f₁` and `f₃` and approximately condition.
ŷ₁, ŷ₃ = rand(rng, [y₁(X₁), y₃(X₃)]);
f₁′, f₂′, f₃′ = (f₁, f₂, f₃) | Titsias([f₁(X₁)←ŷ₁, f₃(X₃)←ŷ₃], [f₁(Z), f₃(Z)], sqrt(σ²));

# Define some plotting stuff.
Np, S = 500, 25;
Xp = range(-2.5, stop=12.5, length=Np);

# Sample jointly from the posterior over each process.
f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp), f₂′(Xp), f₃′(Xp)], S);

# Compute posterior marginals.
μf₁′, σf₁′ = marginals(f₁′(Xp));
μf₂′, σf₂′ = marginals(f₂′(Xp));
μf₃′, σf₃′ = marginals(f₃′(Xp));

@show logpdf([y₁(X₁), y₃(X₃)], [ŷ₁, ŷ₃]);
fb, yb = BlockGP([f₁(X₁), f₃(X₃)]), BlockVector([ŷ₁, ŷ₃]);
@show elbo(fb, yb, fb, sqrt(σ²));

# Perform exact computations.
f′1, f′2, f′3 = (f₁, f₂, f₃) | (y₁(X₁)←ŷ₁, y₃(X₃)←ŷ₃);
f′1Xp, f′2Xp, f′3Xp = rand(rng, [f′1(Xp), f′2(Xp), f′3(Xp)], S);
mf′1, σf′1 = marginals(f′1(Xp));
mf′2, σf′2 = marginals(f′2(Xp));
mf′3, σf′3 = marginals(f′3(Xp));



###########################  Plot results  ###########################

plotly();
approximate_posterior_plot = plot();

# Plot posterior marginal variances
plot!(approximate_posterior_plot, Xp, [μf₁′ μf₁′];
    linewidth=0.0,
    fillrange=[μf₁′ .- 3 .* σf₁′, μf₁′ .+ 3 * σf₁′],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
plot!(approximate_posterior_plot, Xp, [μf₂′ μf₂′];
    linewidth=0.0,
    fillrange=[μf₂′ .- 3 .* σf₂′, μf₂′ .+ 3 * σf₂′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
plot!(approximate_posterior_plot, Xp, [μf₃′ μf₃′];
    linewidth=0.0,
    fillrange=[μf₃′ .- 3 .* σf₃′, μf₃′ .+ 3 * σf₃′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");

# Plot joint posterior samples
plot!(approximate_posterior_plot, Xp, f₁′Xp,
    linecolor=:red,
    linealpha=0.2,
    label="");
plot!(approximate_posterior_plot, Xp, f₂′Xp,
    linecolor=:green,
    linealpha=0.2,
    label="");
plot!(approximate_posterior_plot, Xp, f₃′Xp,
    linecolor=:blue,
    linealpha=0.2,
    label="");

# Plot posterior means
plot!(approximate_posterior_plot, Xp, μf₁′;
    linecolor=:red,
    linewidth=2.0,
    label="f1");
plot!(approximate_posterior_plot, Xp, μf₂′;
    linecolor=:green,
    linewidth=2.0,
    label="f2");
plot!(approximate_posterior_plot, Xp, μf₃′;
    linecolor=:blue,
    linewidth=2.0,
    label="f3");

# Plot observations
scatter!(approximate_posterior_plot, X₁, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
scatter!(approximate_posterior_plot, X₃, ŷ₃;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");

# display(approximate_posterior_plot);

# # savefig(approximate_posterior_plot, "process_decomposition.png")

posterior_plot = plot();

items = [
    (mf′1, σf′1, f′1Xp, :red, "f1"),
    (mf′2, σf′2, f′2Xp, :green, "f2"),
    (mf′3, σf′3, f′3Xp, :blue, "f3"),
];

# Plot posterior marginal variances
for (μ, σ, _, colour, name) in items
    plot!(posterior_plot, Xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="");
end
for (μ, σ, f, colour, name) in items
    plot!(posterior_plot, Xp, f,
        linecolor=colour,
        linealpha=0.2,
        label="");
end
for (μ, σ, f, colour, name) in items
    plot!(posterior_plot, Xp, μ;
        linecolor=colour,
        linewidth=2.0,
        label=name);
end

# Plot observations
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

joint_plot = plot(approximate_posterior_plot, posterior_plot; layout=(2, 1));
display(joint_plot);
