using Stheno, Plots, Random
using Stheno: @model



###########################  Define our model  ###########################

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
@model function model()
    f₁ = GP(ConstantMean(randn()), EQ(), )
    f₂ = GP(EQ(), )
    f₃ = f₁ + f₂
    return f₁, f₂, f₃
end

# Randomly sample `N₁` locations at which to measure `f` using `y1`, and `N2` locations
# at which to measure `f` using `y2`.
rng, N₁, N₃ = MersenneTwister(123546), 10, 11;
X₁, X₃ = sort(rand(rng, N₁) * 10), sort(rand(rng, N₃) * 10);
f₁, f₂, f₃ = model();

# Generate some toy observations of `f₁` and `f₃`.
ŷ₁, ŷ₃ = rand(rng, [f₁(X₁), f₃(X₃)]);

# Compute the posterior processes.
(f₁′, f₂′, f₃′) = (f₁, f₂, f₃) | (f₁(X₁)←ŷ₁, f₃(X₃)←ŷ₃);

# Define some plotting stuff.
Np, S = 500, 25;
Xp = range(-2.5, stop=12.5, length=Np);

# Sample jointly from the posterior over each process.
f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp), f₂′(Xp), f₃′(Xp)], S);

# Compute posterior marginals.
μf₁′, σf₁′ = marginals(f₁′(Xp));
μf₂′, σf₂′ = marginals(f₂′(Xp));
μf₃′, σf₃′ = marginals(f₃′(Xp));



###########################  Plot results  ###########################

plotly();
posterior_plot = plot();

# Plot posterior marginal variances
plot!(posterior_plot, Xp, [μf₁′ μf₁′];
    linewidth=0.0,
    fillrange=[μf₁′ .- 3 .* σf₁′, μf₁′ .+ 3 * σf₁′],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
plot!(posterior_plot, Xp, [μf₂′ μf₂′];
    linewidth=0.0,
    fillrange=[μf₂′ .- 3 .* σf₂′, μf₂′ .+ 3 * σf₂′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
plot!(posterior_plot, Xp, [μf₃′ μf₃′];
    linewidth=0.0,
    fillrange=[μf₃′ .- 3 .* σf₃′, μf₃′ .+ 3 * σf₃′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");

# Plot joint posterior samples
plot!(posterior_plot, Xp, f₁′Xp,
    linecolor=:red,
    linealpha=0.2,
    label="");
plot!(posterior_plot, Xp, f₂′Xp,
    linecolor=:green,
    linealpha=0.2,
    label="");
plot!(posterior_plot, Xp, f₃′Xp,
    linecolor=:blue,
    linealpha=0.2,
    label="");

# Plot posterior means
plot!(posterior_plot, Xp, μf₁′;
    linecolor=:red,
    linewidth=2.0,
    label="f1");
plot!(posterior_plot, Xp, μf₂′;
    linecolor=:green,
    linewidth=2.0,
    label="f2");
plot!(posterior_plot, Xp, μf₃′;
    linecolor=:blue,
    linewidth=2.0,
    label="f3");

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

display(posterior_plot);
