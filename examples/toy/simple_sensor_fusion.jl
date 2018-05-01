using Stheno, Plots
plotly();



###########################  Define and inspect our model  ###########################

#=
In this example, `f` is an unknown real-valued function that we wish to infer. To achieve
this, we have access to two sensors. The first returns noisy estimates of `f`, where we have
been reliably informed by whoever designed the sensor that the mean of the noise is given by
`sin(x) - 5 + sqrt(abs(x))`, and that it's variance is low (1e-2). (how the designer
estimated this function, is why a sensor might possibly have such a weird mean error is
beyond the scope of this example) The second returns biased measurements of `f`, where the
bias is known to be 3.5. The model below specifies a model for this scenario.
=#
function model(gpc)

    # Define a smooth latent process that we wish to infer.
    f = GP(ZeroMean{Float64}(), EQ(), gpc)

    # Define the two noise processes described.
    noise1 = GP(CustomMean(x->sin.(x) .- 5.0 .+ sqrt.(abs.(x))), Noise(1e-2), gpc)
    noise2 = GP(ConstantMean(3.5), Noise(1e-1), gpc)

    # Define the processes that we get to observe.
    y1 = f + noise1
    y2 = f + noise2

    return f, noise1, noise2, y1, y2
end

# Randomly sample `N1` locations at which to measure `f` using `y1`, and `N2` locations
# at which to measure `f` using `y2`.
rng, N1, N2 = MersenneTwister(123546), 3, 10;
X1, X2 = sort(rand(rng, N1) * 10), sort(rand(rng, N2) * 10);
f, noise1, noise2, y1, y2 = model(GPC());

# Generate some toy observations of `y1` and `y2`.
ŷ1, ŷ2 = rand(rng, [y1, y2], [X1, X2]);

# Compute the posterior process.
(f′, y1′, y2′) = (f, y1, y2) | (y1(X1)←ŷ1, y2(X2)←ŷ2);

# Define some plotting stuff.
Nplot, S = 500, 100;
Xp = linspace(-2.5, 12.5, Nplot);

# Sample from posterior and write directly to file.
f′Xp = rand(rng, f′, Xp, S);
y1′Xp = rand(rng, y1′, Xp, S);
y2′Xp = rand(rng, y2′, Xp, S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf′, σf′ = mean(f′, Xp), marginal_std(f′, Xp);
μy1′, σy1′ = mean(y1′, Xp), marginal_std(y1′, Xp);
μy2′, σy2′ = mean(y2′, Xp), marginal_std(y2′, Xp);



###########################  Plot results - USE ONLY Julia-0.6!  ###########################

posterior_plot = plot();

# Plot posterior of first noise process.
plot!(posterior_plot, Xp, μy1′;
    linecolor=:red,
    linewidth=2.0,
    label="");
plot!(posterior_plot, Xp, [μy1′ μy1′];
    linewidth=0.0,
    fillrange=[μy1′ .- 3 .* σy1′, μy1′ .+ 3 * σy1′],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
scatter!(posterior_plot, X1, ŷ1;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 1");
scatter!(posterior_plot, Xp, y1′Xp,
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="");

# Plot posterior of second noise process.
plot!(posterior_plot, Xp, μy2′;
    linecolor=:green,
    linewidth=2.0,
    label="");
plot!(posterior_plot, Xp, [μy2′ μy2′];
    linewidth=0.0,
    fillrange=[μy2′ .- 3 .* σy2′, μy2′ .+ 3 * σy2′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
scatter!(posterior_plot, X2, ŷ2;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.8,
    label="Sensor 2");
scatter!(posterior_plot, Xp, y2′Xp,
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=0.5,
    markeralpha=0.3,
    label="");

# Plot posterior of smooth latent function.
plot!(posterior_plot, Xp, f′Xp;
    linecolor=:blue,
    linealpha=0.2,
    label="");
plot!(posterior_plot, Xp, μf′;
    linecolor=:blue,
    linewidth=2.0,
    label="");
plot!(posterior_plot, Xp, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′.- 3  .* σf′ μf′ .+ 3 .* σf′],
    fillalpha=0.5,
    fillcolor=:blue,
    label="");


display(posterior_plot);
