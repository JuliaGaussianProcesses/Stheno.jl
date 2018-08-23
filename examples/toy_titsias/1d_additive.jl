using Stheno, Plots, Random
plotly();

# Define model.
σ_noise, ω, T = 1e0, 1.0, 25.0;
gpc = GPC();
f₁ = GP(periodic(EQ(), ω), gpc);
f₂ = GP(scale(EQ(), 0.1), gpc);
f₃ = f₁ + f₂;
y = f₃ + GP(Noise(σ_noise^2), gpc);

# Sample from marginal process to generate toy data.
rng, S = MersenneTwister(123456), 25;
X, Xp = linspace(0.0, T, 100), linspace(-2.5, T + 2.5, 500);
ŷ = rand(rng, y(X));


# Plots for stuff.
f₁′_plot, f₂′_plot, f₃′_plot = plot(), plot(), plot();

posterior_plot = plot();
approx_in_marginal_posterior_plot = plot();
approx_in_latents_posterior_plot = plot();



##################################### Exact Inference ######################################

# Compute the posterior processes, sample from them, and compute marginals.
@show logpdf(y(X), ŷ);
f₁′, f₂′, f₃′ = (f₁, f₂, f₃) | (y(X) ← ŷ);
f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp), f₂′(Xp), f₃′(Xp)], S);
μ₁′, σ₁′ = marginals(f₁′(Xp));
μ₂′, σ₂′ = marginals(f₂′(Xp));
μ₃′, σ₃′ = marginals(f₃′(Xp));

items = [
    (μ₁′, σ₁′, f₁′Xp, :red, "exact", f₁′_plot),
    (μ₂′, σ₂′, f₂′Xp, :red, "", f₂′_plot),
    (μ₃′, σ₃′, f₃′Xp, :red, "", f₃′_plot),
];

# Posterior marginal variance.
for (μ, σ, _, colour, name, plt) in items
    plot!(plt, Xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="");
end

# Posterior samples.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, Xp, f,
        linecolor=colour,
        linealpha=0.2,
        label="");
end

# Posterior mean.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, Xp, μ;
        linecolor=colour,
        linewidth=2.0,
        label=name);
end

# Plot observations.
scatter!(posterior_plot, X, ŷ;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");



##################### Approximate inference with pseudo-data in f₃ #########################

# Compute approximate posterior process.
M₃ = 15;
Z₃ = linspace(0.0, T, M₃);
Z₃ = 0:T
Z₃ = 0:0.5:T
# Z₃ = 0:0.25:
u₃ = f₃(Z₃);
μ′u₃, Σ′u₃ = Stheno.optimal_q(f₃(X), ŷ, u₃, σ_noise);
q_u₃ = Stheno.Titsias(u₃, μ′u₃, Σ′u₃, gpc);
f₁′u₃, f₂′u₃, f₃′u₃ = (f₁, f₂, f₃) | q_u₃;
f₁′u₃Xp, f₂′u₃Xp, f₃′u₃Xp = rand(rng, [f₁′u₃(Xp), f₂′u₃(Xp), f₃′u₃(Xp)], S);
μ₁′u₃, σ₁′u₃ = marginals(f₁′u₃(Xp));
μ₂′u₃, σ₂′u₃ = marginals(f₂′u₃(Xp));
μ₃′u₃, σ₃′u₃ = marginals(f₃′u₃(Xp));

@show elbo(f₃(X), ŷ, u₃, σ_noise);

items = [
    (μ₁′u₃, σ₁′u₃, f₁′u₃Xp, :green, "Z in f3", f₁′_plot),
    (μ₂′u₃, σ₂′u₃, f₂′u₃Xp, :green, "", f₂′_plot),
    (μ₃′u₃, σ₃′u₃, f₃′u₃Xp, :green, "", f₃′_plot),
];

# Posterior marginal variance.
for (μ, σ, _, colour, name, plt) in items
    plot!(plt, Xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="");
end

# # Posterior samples.
# for (μ, σ, f, colour, name, plt) in items
#     plot!(plt, Xp, f,
#         linecolor=colour,
#         linealpha=0.2,
#         label="");
# end

# Posterior mean.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, Xp, μ;
        linecolor=colour,
        linewidth=2.0,
        label=name);
end

# Plot observations and pseudo-input locations.
scatter!(approx_in_marginal_posterior_plot, X, ŷ;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
scatter!(approx_in_marginal_posterior_plot, Z₃, zeros(M₃);
    markercolor=:black,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z");



############# Perform approximate inference by placing pseudo-data in f₁ and f₂ ############

# Compute approximate posterior process.
M₁, M₂ = 10, 5;
Z₁, Z₂ = linspace(0.0, 1 / ω, M₁), linspace(0.0, T, M₂);
u₁₂ = BlockGP([f₁(Z₁), f₂(Z₂)]);
μ′u, Σ′u = Stheno.optimal_q(f₃(X), ŷ, u₁₂, σ_noise);
conditioner = Stheno.Titsias(u₁₂, μ′u, Σ′u, gpc);
f₁′u₁₂, f₂′u₁₂, f₃′u₁₂ = (f₁, f₂, f₃) | conditioner;

f₁′u₁₂Xp, f₂′u₁₂Xp, f₃′u₁₂Xp = rand(rng, [f₁′u₁₂(Xp), f₂′u₁₂(Xp), f₃′u₁₂(Xp)], S);
μ₁′u₁₂, σ₁′u₁₂ = marginals(f₁′u₁₂(Xp));
μ₂′u₁₂, σ₂′u₁₂ = marginals(f₂′u₁₂(Xp));
μ₃′u₁₂, σ₃′u₁₂ = marginals(f₃′u₁₂(Xp));

@show elbo(f₃(X), ŷ, u₁₂, σ_noise);

items = [
    (μ₁′u₁₂, σ₁′u₁₂, f₁′u₁₂Xp, :blue, "Z in f1 and f2", f₁′_plot),
    (μ₂′u₁₂, σ₂′u₁₂, f₂′u₁₂Xp, :blue, "", f₂′_plot),
    (μ₃′u₁₂, σ₃′u₁₂, f₃′u₁₂Xp, :blue, "", f₃′_plot),
];

# Posterior marginal variance.
for (μ, σ, _, colour, name, plt) in items
    plot!(plt, Xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="");
end

# # Posterior samples.
# for (μ, σ, f, colour, name, plt) in items
#     plot!(plt, Xp, f,
#         linecolor=colour,
#         linealpha=0.2,
#         label="");
# end

# Posterior mean.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, Xp, μ;
        linecolor=colour,
        linewidth=2.0,
        label=name);
end

# Plot observations and pseudo-input locations.
scatter!(approx_in_latents_posterior_plot, X, ŷ;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
scatter!(approx_in_latents_posterior_plot, Z₁, zeros(M₁);
    markercolor=:black,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z₁");
scatter!(approx_in_latents_posterior_plot, Z₂, zeros(M₂);
    markercolor=:magenta,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z₂");



# display(posterior_plot);
# display(approx_in_marginal_posterior_plot);
# display(approx_in_latents_posterior_plot);

plot(f₁′_plot, f₂′_plot, f₃′_plot; layout=(3, 1))
