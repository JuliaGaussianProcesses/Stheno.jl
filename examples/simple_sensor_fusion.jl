using Revise
using Stheno, Random, DelimitedFiles



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
observations = rand(rng, [y1, y2], [X1, X2]);
ŷ1, ŷ2 = observations[1:N1], observations[N1+1:N1+N2];

# Compute the posterior process.
(f′, y1′, y2′) = (f, y1, y2) | (y1(X1)←ŷ1, y2(X2)←ŷ2);

# Define some plotting stuff.
Nplot, S = 500, 100;
Xplot = range(-2.5, stop=12.5, length=Nplot);

# Sample from posterior and write directly to file.
open("simple_sensor_fusion/posterior_samples.csv", "w") do io
    writedlm(io, rand(rng, f′, Xplot, S), ',')
end

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf′, σf′ = mean(f′, Xplot), marginal_std(f′, Xplot);
μy1′, σy1′ = mean(y1′, Xplot), marginal_std(y1′, Xplot);
μy2′, σy2′ = mean(y2′, Xplot), marginal_std(y2′, Xplot);
open("simple_sensor_fusion/posterior_marginals.csv", "w") do io
    writedlm(io, hcat(Xplot, μf′, σf′, μy1′, σy1′, μy2′, σy2′), ',')
end

# Write observations to file.
open("simple_sensor_fusion/observations_1.csv", "w") do io
    writedlm(io, hcat(X1, ŷ1), ',')
end
open("simple_sensor_fusion/observations_2.csv", "w") do io
    writedlm(io, hcat(X2, ŷ2), ',')
end



###########################  Plot results - USE ONLY Julia-0.6!  ###########################

# Only run me from Julia-0.6. Plots doesn't work on 0.7 at the minute.
using Plots
plotly();

# Load posterior predictive quantities.
f′Xp = readcsv("simple_sensor_fusion/posterior_samples.csv");
posterior_marginals = readcsv("simple_sensor_fusion/posterior_marginals.csv");
obs1 = readcsv("simple_sensor_fusion/observations_1.csv");
obs2 = readcsv("simple_sensor_fusion/observations_2.csv");

Xp = posterior_marginals[:, 1]
μf′, σf′ = posterior_marginals[:, 2], posterior_marginals[:, 3];
μy1′, σy1′ = posterior_marginals[:, 4], posterior_marginals[:, 5];
μy2′, σy2′ = posterior_marginals[:, 6], posterior_marginals[:, 7];
X1, ŷ1 = obs1[:, 1], obs1[:, 2];
X2, ŷ2 = obs2[:, 1], obs2[:, 2];

# Visualise posterior distribution and observational data.
posterior_plot = plot(Xp, f′Xp;
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
    markeralpha=0.7,
    label="Sensor 1");

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
    markeralpha=0.7,
    label="Sensor 2");
display(posterior_plot);



