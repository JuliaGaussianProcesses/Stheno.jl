using Revise
using Stheno, Random, DelimitedFiles



###########################  Define and inspect our model  ###########################

# Define the vanilla GP-regression generative model.
function model(gpc)

    # Define a smooth latent process.
    f = GP(ZeroMean{Float64}(), EQ(), gpc)

    # Define a latent noise process.
    noise = GP(ZeroMean{Float64}(), Noise(1e-2), gpc)

    # Sum them to get the process of which we shall make observations.
    y = f + noise

    # Return all three processes so that we can inspect all of them.
    return f, noise, y
end

# Generate some input locations and sample from the prior.
rng, N = MersenneTwister(123546), 500;
X = sort(rand(rng, N) * 10);
f, noise, y = model(GPC());

# Take a look at the mean function and kernel of y.
@show mean(y), kernel(y);

# Take a look at the cross-covariance between f and y.
@show kernel(f, y);

# Inspect the cross-covariance between noise and y.
@show kernel(noise, y);



###########################  Sample jointly from the prior  ###########################

# Sample jointly from the prior distribution over the three processes.
out = rand(rng, [f, noise, y], [X, X, X]);
fX, noiseX, yX = out[1:N], out[N+1:2N], out[2N+1:end];

# Write results to file and run from Julia-0.6 because 0.7 can't plot anything yet...
open("vanilla_noisy_regression/prior.csv", "w") do io
    writedlm(io, hcat(X, fX, noiseX, yX), ',')
end



#######################  Do posterior inference give a few samples  #######################

# Sample a few points from the prior and compute the posterior processes.
N = 15;
X = sort(rand(rng, N) * 10);
ŷ = rand(rng, y, X);
f′, noise′, y′ = (f, noise, y) | (y(X) ← ŷ);

# There appear to be some substantial numerical problems associated with generating samples
# jointly from all three processes, thus only f′ and noise′ are considered.
Nplot, S = 500, 100;
Xplot = range(-2.0, stop=12.0, length=Nplot);
out = rand(rng, [f′, noise′], [Xplot, Xplot], S);

open("vanilla_noisy_regression/posterior.csv", "w") do io
    writedlm(io, hcat(Xplot, out[1:Nplot, :], out[Nplot+1:end, :]), ',')
end
open("vanilla_noisy_regression/observations.csv", "w") do io
    writedlm(io, hcat(X, ŷ), ',')
end

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf′, σf′, σy′ = mean(f′, Xplot), marginal_std(f′, Xplot), marginal_std(y′, Xplot);

open("vanilla_noisy_regression/posterior_marginals.csv", "w") do io
    writedlm(io, hcat(μf′, σf′, σy′), ',')
end




###########################  Plot results - USE ONLY Julia-0.6!  ###########################

# Only run me from Julia-0.6. Plots doesn't work on 0.7 at the minute.
using Plots
plotly();

# Plot results from prior sampling.
data = readcsv("vanilla_noisy_regression/prior.csv");
X, fX, noiseX, yX = data[:, 1], data[:, 2], data[:, 3], data[:, 4];

prior_plot = plot(X, fX, label="f");
scatter!(a, X, noiseX,
    label="noise",
    markershape=:xcross,
    markerstrokewidth=0.0,
    markersize=2);
scatter!(prior_plot, X, yX,
    label="y",
    markershape=:xcross,
    markerstrokewidth=0.0,
    markersize=2);
plot!(prior_plot, X, fX + noiseX - yX,
    label="f + noise - y",
    linecolor=:red);

# Plot results from posterior inference.
data_post = readcsv("vanilla_noisy_regression/posterior.csv");
marginals = readcsv("vanilla_noisy_regression/posterior_marginals.csv");
obs = readcsv("vanilla_noisy_regression/observations.csv");
S = 100;
Xplot, f′Xplot, noise′Xplot = data_post[:, 1], data_post[:, 2:S+1], data_post[:, S+2:end];
Xobs, yobs = obs[:, 1], obs[:, 2];
μf′, σf′, σy′ = marginals[:, 1], marginals[:, 2], marginals[:, 3];

posterior_plot = plot(Xplot, f′Xplot;
    linecolor=:blue,
    linealpha=0.2,
    legend=false);
plot!(posterior_plot, Xplot, μf′;
    linewidth=2.0,
    linecolor=:blue);
plot!(posterior_plot, Xplot, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′ .- 3 .* σy′, μf′ .+ 3 * σy′],
    fillalpha=0.3,
    fillcolor=:red);
plot!(posterior_plot, Xplot, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′.- 3  .* σf′ μf′ .+ 3 .* σf′],
    fillalpha=0.5,
    fillcolor=:blue);
scatter!(posterior_plot, Xobs, yobs;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    legend=false);
joint_plot = plot(prior_plot, posterior_plot, layout=(2, 1));
display(joint_plot);
