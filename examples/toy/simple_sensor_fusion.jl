using Stheno, Random, Plots
using Stheno: @model

###########################  Define and inspect our model  ###########################

rng = MersenneTwister(123456);

#=
In this example, `f` is an unknown real-valued function that we wish to infer. To achieve
this, we have access to two sensors. The first returns noisy estimates of `f`, where we have
been reliably informed by whoever designed the sensor that the mean of the noise is given by
`sin(x) - 5 + sqrt(abs(x))`, and that it's variance is low (1e-2). (how the designer
estimated this function, is why a sensor might possibly have such a weird mean error is
beyond the scope of this example) The second returns biased measurements of `f`, where the
bias is known to be 3.5. The model below specifies a model for this scenario.
=#
@model function model()

    # Define a smooth latent process that we wish to infer.
    f = GP(EQ())

    # Define the two noise processes described.
    noise1 = GP(CustomMean(x->sin.(x) .- 5.0 .+ sqrt.(abs.(x))), Noise(1e-2))
    noise2 = GP(ConstantMean(3.5), Noise(1e-1))

    # Define the processes that we get to observe.
    y1 = f + noise1
    y2 = f + noise2

    return f, noise1, noise2, y1, y2
end
f, noise₁, noise₂, y₁, y₂ = model();

# Generate some toy observations of `y1` and `y2`.
X₁, X₂ = sort(rand(rng, 3) * 10), sort(rand(rng, 10) * 10);
ŷ₁, ŷ₂ = rand(rng, [y₁(X₁), y₂(X₂)]);

# Compute the posterior processes.
(f′, y₁′, y₂′) = (f, y₁, y₂) | (y₁(X₁)←ŷ₁, y₂(X₂)←ŷ₂);

# Sample jointly from the posterior processes and compute posterior marginals.
Xp = range(-2.5, stop=12.5, length=500);
f′Xp, y₁′Xp, y₂′Xp = rand(rng, [f′(Xp), y₁′(Xp), y₂′(Xp)], 100);
μf′, σf′ = marginals(f′(Xp));
μy₁′, σy₁′ = marginals(y₁′(Xp));
μy₂′, σy₂′ = marginals(y₂′(Xp));



###########################  Plot results  ###########################

plotly();

posterior_plot = plot();

# Plot posterior marginal std. dev.
plot!(posterior_plot, Xp, [μy₁′ μy₁′];
    linewidth=0.0,
    fillrange=[μy₁′ .- 3 .* σy₁′, μy₁′ .+ 3 * σy₁′],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
plot!(posterior_plot, Xp, [μy₂′ μy₂′];
    linewidth=0.0,
    fillrange=[μy₂′ .- 3 .* σy₂′, μy₂′ .+ 3 * σy₂′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
plot!(posterior_plot, Xp, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′.- 3  .* σf′ μf′ .+ 3 .* σf′],
    fillalpha=0.5,
    fillcolor=:blue,
    label="");

# Plot posterior marginal samples.
scatter!(posterior_plot, Xp, y₁′Xp,
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="");
scatter!(posterior_plot, Xp, y₂′Xp,
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="");
plot!(posterior_plot, Xp, f′Xp;
    linecolor=:blue,
    linealpha=0.2,
    label="");

# Plot posterior means
plot!(posterior_plot, Xp, μy₁′;
    linecolor=:red,
    linewidth=2.0,
    label="");
plot!(posterior_plot, Xp, μy₂′;
    linecolor=:green,
    linewidth=2.0,
    label="");
plot!(posterior_plot, Xp, μf′;
    linecolor=:blue,
    linewidth=2.0,
    label="Latent Function");

# Plot posterior of first noise process.
scatter!(posterior_plot, X₁, ŷ₁;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 1");
scatter!(posterior_plot, X₂, ŷ₂;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 2");


display(posterior_plot);
