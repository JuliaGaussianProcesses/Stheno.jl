# Please run from the `basic_gppp` directory.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Random, Plots, Statistics
using Stheno: @model, EQ, Noise

###########################  Define and inspect our model  ###########################

rng = MersenneTwister(123456);

#=
In this example, `f` is an unknown real-valued function that we wish to infer. To achieve
this, we have access to two sensors. The first returns noisy estimates of `f`, where we have
been reliably informed by whoever designed the sensor that the mean of the noise is given by
`sin(x) - 5 + sqrt(abs(x))`, and that it's variance is low (1e-2). How the designer
estimated this function, and why a sensor might possibly have such a strange mean error, is
beyond the scope of this example. The second returns biased measurements of `f`, where the
bias is known to be 3.5. The model below specifies a model for this scenario.
=#
@model function model()

    # Define a smooth latent process that we wish to infer.
    f = GP(EQ())

    # Define the two noise processes described.
    noise1 = sqrt(1e-2) * GP(Noise()) + (x->sin.(x) .- 5.0 .+ sqrt.(abs.(x)))
    noise2 = sqrt(1e-1) * GP(3.5, Noise())

    # Define the processes that we get to observe.
    y1 = f + noise1
    y2 = f + noise2

    return f, noise1, noise2, y1, y2
end
f, noise₁, noise₂, y₁, y₂ = model();

# Generate some toy observations of `y₁` and `y₂`.
x₁, x₂ = sort(rand(rng, 3) * 10), sort(rand(rng, 10) * 10);
ŷ₁, ŷ₂ = rand(rng, [y₁(x₁), y₂(x₂)]);

# Compute the posterior processes.
(f′, y₁′, y₂′) = (f, y₁, y₂) | (y₁(x₁)←ŷ₁, y₂(x₂)←ŷ₂);

# Sample jointly from the posterior processes and compute posterior marginals.
xp = range(-2.5, stop=12.5, length=500);
f′xp, y₁′xp, y₂′xp = rand(rng, [f′(xp, 1e-9), y₁′(xp, 1e-9), y₂′(xp, 1e-9)], 100);

ms₁ = marginals(f′(xp));
ms₂ = marginals(y₁′(xp));
ms₃ = marginals(y₂′(xp));

μf′, σf′ = mean.(ms₁), std.(ms₁)
μy₁′, σy₁′ = mean.(ms₂), std.(ms₂)
μy₂′, σy₂′ = mean.(ms₃), std.(ms₃)



###########################  Plot results  ###########################

gr();

posterior_plot = plot();

# Plot posterior marginal std. dev.
plot!(posterior_plot, xp, [μy₁′ μy₁′];
    linewidth=0.0,
    fillrange=[μy₁′ .- 3 .* σy₁′, μy₁′ .+ 3 * σy₁′],
    fillalpha=0.3,
    fillcolor=:red,
    label="",
);
plot!(posterior_plot, xp, [μy₂′ μy₂′];
    linewidth=0.0,
    fillrange=[μy₂′ .- 3 .* σy₂′, μy₂′ .+ 3 * σy₂′],
    fillalpha=0.3,
    fillcolor=:green,
    label="",
);
plot!(posterior_plot, xp, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′.- 3  .* σf′ μf′ .+ 3 .* σf′],
    fillalpha=0.5,
    fillcolor=:blue,
    label="",
);

# Plot posterior marginal samples.
scatter!(posterior_plot, xp, y₁′xp,
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="",
);
scatter!(posterior_plot, xp, y₂′xp,
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="",
);
plot!(posterior_plot, xp, f′xp;
    linecolor=:blue,
    linealpha=0.2,
    label="",
);

# Plot posterior means.
plot!(posterior_plot, xp, μy₁′;
    linecolor=:red,
    linewidth=2.0,
    label="",
);
plot!(posterior_plot, xp, μy₂′;
    linecolor=:green,
    linewidth=2.0,
    label="",
);
plot!(posterior_plot, xp, μf′;
    linecolor=:blue,
    linewidth=2.0,
    label="Latent Function",
);

# Plot samples on which we conditioned.
scatter!(posterior_plot, x₁, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 1",
);
scatter!(posterior_plot, x₂, ŷ₂;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 2",
);

display(posterior_plot);
