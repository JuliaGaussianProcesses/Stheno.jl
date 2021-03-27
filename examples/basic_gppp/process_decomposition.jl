# Please run from the `basic_gppp` directory.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random, Plots, Stheno
using Stheno: @model, SEKernel



###########################  Define our model  ###########################

# Define a distribution over f₁, f₂, and f₃, where f₃(x) = f₁(x) + f₂(x).
@model function model()
    f₁ = GP(randn(), SEKernel())
    f₂ = GP(SEKernel())
    f₃ = f₁ + f₂
    return f₁, f₂, f₃
end

# Construct the processes in the model.
f₁, f₂, f₃ = model()

# Randomly sample `N₁` and `N₃` locations at which to observe `f₁` and `f₃` respectively.
rng, N₁, N₃ = MersenneTwister(123546), 10, 11
x₁, x₃ = sort(rand(rng, N₁) * 10), sort(rand(rng, N₃) * 10)

# Generate some toy of `f₁` and `f₃`, `y₁` and `y₃` respectively.
y₁, y₃ = rand(rng, [f₁(x₁), f₃(x₃)])

# Compute the posterior processes.
(f₁_post, f₂_post, f₃_post) = (f₁, f₂, f₃) | (f₁(x₁)←y₁, f₃(x₃)←y₃)

# Define some plotting stuff.
Np, S = 500, 25;
xp = range(-2.5, stop=12.5, length=Np);

# Sample jointly from the posterior over each process.
f₁_post_xp, f₂_post_xp, f₃_post_xp =
    rand(rng, [f₁_post(xp, 1e-9), f₂_post(xp, 1e-9), f₃_post(xp, 1e-9)], S);

# Compute posterior marginals.
ms₁ = marginals(f₁_post(xp));
ms₂ = marginals(f₂_post(xp));
ms₃ = marginals(f₃_post(xp));

μf₁′, σf₁′ = mean.(ms₁), std.(ms₁);
μf₂′, σf₂′ = mean.(ms₂), std.(ms₂);
μf₃′, σf₃′ = mean.(ms₃), std.(ms₃);



###########################  Plot results  ###########################

gr();
posterior_plot = plot();

# Plot posterior marginal variances
plot!(posterior_plot, xp, [μf₁′ μf₁′];
    linewidth=0.0,
    fillrange=[μf₁′ .- 3 .* σf₁′, μf₁′ .+ 3 * σf₁′],
    fillalpha=0.3,
    fillcolor=:red,
    label="",
);
plot!(posterior_plot, xp, [μf₂′ μf₂′];
    linewidth=0.0,
    fillrange=[μf₂′ .- 3 .* σf₂′, μf₂′ .+ 3 * σf₂′],
    fillalpha=0.3,
    fillcolor=:green,
    label="",
);
plot!(posterior_plot, xp, [μf₃′ μf₃′];
    linewidth=0.0,
    fillrange=[μf₃′ .- 3 .* σf₃′, μf₃′ .+ 3 * σf₃′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="",
);

# Plot joint posterior samples
plot!(posterior_plot, xp, f₁_post_xp,
    linecolor=:red,
    linealpha=0.2,
    label="",
);
plot!(posterior_plot, xp, f₂_post_xp,
    linecolor=:green,
    linealpha=0.2,
    label="",
);
plot!(posterior_plot, xp, f₃_post_xp,
    linecolor=:blue,
    linealpha=0.2,
    label="",
);

# Plot posterior means
plot!(posterior_plot, xp, μf₁′;
    linecolor=:red,
    linewidth=2.0,
    label="f1",
);
plot!(posterior_plot, xp, μf₂′;
    linecolor=:green,
    linewidth=2.0,
    label="f2",
);
plot!(posterior_plot, xp, μf₃′;
    linecolor=:blue,
    linewidth=2.0,
    label="f3",
);

# Plot observations
scatter!(posterior_plot, x₁, y₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
scatter!(posterior_plot, x₃, y₃;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

display(posterior_plot);
