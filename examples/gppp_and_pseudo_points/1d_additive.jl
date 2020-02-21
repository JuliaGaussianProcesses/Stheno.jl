# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Plots, Random
gr()

# Define model.
σ², ω, T = 1e0, 1.0, 25.0
gpc = GPC()
f₁ = periodic(GP(EQ(), gpc), ω)
f₂ = GP(0.1 * EQ(), gpc)
f₃ = f₁ + f₂

# Sample from marginal process to generate toy data.
rng = MersenneTwister(123456)
S = 25
x = range(0.0, T; length=100)
xp = range(-2.5, T + 2.5; length=500)
fx = f₃(x, σ²)
y = rand(rng, fx)

# Plots for stuff.
f₁′_plot, f₂′_plot, f₃′_plot = plot(), plot(), plot();

posterior_plot = plot();
approx_in_marginal_posterior_plot = plot();
approx_in_latents_posterior_plot = plot();



##################################### Exact Inference ######################################

# Compute the posterior processes, sample from them, and compute marginals.
@show logpdf(fx, y)
f₁′, f₂′, f₃′ = (f₁, f₂, f₃) | (fx ← y)
f₁′xp, f₂′xp, f₃′xp = rand(rng, [f₁′(xp, 1e-6), f₂′(xp, 1e-6), f₃′(xp, 1e-6)], S)

ms₁ = marginals(f₁′(xp))
ms₂ = marginals(f₂′(xp))
ms₃ = marginals(f₃′(xp))
μ₁′, σ₁′ = mean.(ms₁), std.(ms₁)
μ₂′, σ₂′ = mean.(ms₂), std.(ms₂)
μ₃′, σ₃′ = mean.(ms₃), std.(ms₃)

items = [
    (μ₁′, σ₁′, f₁′xp, :red, "exact", f₁′_plot),
    (μ₂′, σ₂′, f₂′xp, :red, "", f₂′_plot),
    (μ₃′, σ₃′, f₃′xp, :red, "", f₃′_plot),
]

# Posterior marginal variance.
for (μ, σ, _, colour, name, plt) in items
    plot!(plt, xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="",
    );
end

# Posterior samples.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, xp, f,
        linecolor=colour,
        linealpha=0.2,
        label="",
    );
end

# Posterior mean.
for (μ, σ, f, colour, name, plt) in items
    plot!(plt, xp, μ;
        linecolor=colour,
        linewidth=2.0,
        label=name,
    );
end

# Plot observations.
scatter!(posterior_plot, x, y;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);



##################### Approximate inference with pseudo-data in f₃ #########################

# Compute approximate posterior process.
M₃ = 15;
Z₃ = range(0.0, T; length=M₃);
Z₃ = 0:T
Z₃ = 0:0.5:T
# Z₃ = 0:0.25:
u₃ = f₃(Z₃);
# μ′u₃, Σ′u₃ = Stheno.optimal_q(f₃(X), ŷ, u₃, σ_noise);
# q_u₃ = Stheno.Titsias(u₃, μ′u₃, Σ′u₃, gpc);
pseudo_obs = Stheno.PseudoObs(f₃(x) ← y, u₃)
f₁′u₃ = f₁ | pseudo_obs
f₂′u₃ = f₂ | pseudo_obs
f₃′u₃ = f₃ | pseudo_obs
f₁′u₃xp, f₂′u₃xp, f₃′u₃xp = rand(rng, [f₁′u₃(xp), f₂′u₃(xp), f₃′u₃(xp)], S);

μ₁′u₃, σ₁′u₃ = mean.(marginals(f₁′u₃(Xp))), std.(marginals(f₁′u₃(Xp)))
μ₂′u₃, σ₂′u₃ = mean.(marginals(f₂′u₃(Xp))), std.(marginals(f₂′u₃(Xp)))
μ₃′u₃, σ₃′u₃ = mean.(marginals(f₃′u₃(Xp))), std.(marginals(f₃′u₃(Xp)))

@show elbo(fx, y, u₃);

items = [
    (μ₁′u₃, σ₁′u₃, f₁′u₃xp, :green, "Z in f3", f₁′_plot),
    (μ₂′u₃, σ₂′u₃, f₂′u₃xp, :green, "", f₂′_plot),
    (μ₃′u₃, σ₃′u₃, f₃′u₃xp, :green, "", f₃′_plot),
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
        label=name,
    );
end

# Plot observations and pseudo-input locations.
scatter!(approx_in_marginal_posterior_plot, X, ŷ;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
scatter!(approx_in_marginal_posterior_plot, Z₃, zeros(M₃);
    markercolor=:black,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z",
);



############# Perform approximate inference by placing pseudo-data in f₁ and f₂ ############

# Compute approximate posterior process.
M₁, M₂ = 10, 5
Z₁, Z₂ = linspace(0.0, 1 / ω, M₁), linspace(0.0, T, M₂)
u₁₂ = BlockGP([f₁(Z₁), f₂(Z₂)])
μ′u, Σ′u = Stheno.optimal_q(f₃(X), ŷ, u₁₂, σ_noise)
conditioner = Stheno.Titsias(u₁₂, μ′u, Σ′u, gpc)
f₁′u₁₂, f₂′u₁₂, f₃′u₁₂ = (f₁, f₂, f₃) | conditioner

f₁′u₁₂Xp, f₂′u₁₂Xp, f₃′u₁₂Xp = rand(rng, [f₁′u₁₂(Xp), f₂′u₁₂(Xp), f₃′u₁₂(Xp)], S)
μ₁′u₁₂, σ₁′u₁₂ = marginals(f₁′u₁₂(Xp))
μ₂′u₁₂, σ₂′u₁₂ = marginals(f₂′u₁₂(Xp))
μ₃′u₁₂, σ₃′u₁₂ = marginals(f₃′u₁₂(Xp))

@show elbo(f₃(X), ŷ, u₁₂, σ_noise)

items = [
    (μ₁′u₁₂, σ₁′u₁₂, f₁′u₁₂Xp, :blue, "Z in f1 and f2", f₁′_plot),
    (μ₂′u₁₂, σ₂′u₁₂, f₂′u₁₂Xp, :blue, "", f₂′_plot),
    (μ₃′u₁₂, σ₃′u₁₂, f₃′u₁₂Xp, :blue, "", f₃′_plot),
]

# Posterior marginal variance.
for (μ, σ, _, colour, name, plt) in items
    plot!(plt, Xp, [μ, μ];
        linewidth=0.0,
        fillrange=[μ .- 3 .* σ, μ .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=colour,
        label="",
    )
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
        label=name,
    )
end

# Plot observations and pseudo-input locations.
scatter!(approx_in_latents_posterior_plot, X, ŷ;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
scatter!(approx_in_latents_posterior_plot, Z₁, zeros(M₁);
    markercolor=:black,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z₁",
);
scatter!(approx_in_latents_posterior_plot, Z₂, zeros(M₂);
    markercolor=:magenta,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Z₂",
);



# display(posterior_plot);
# display(approx_in_marginal_posterior_plot);
# display(approx_in_latents_posterior_plot);

plot(f₁′_plot, f₂′_plot, f₃′_plot; layout=(3, 1))
