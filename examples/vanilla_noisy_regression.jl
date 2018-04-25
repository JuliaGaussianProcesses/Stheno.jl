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
N = 10;
X = sort(rand(rng, N) * 10);
ŷ = rand(rng, y, X);
f′, noise′, y′ = (f, noise, y) | (y(X) ← ŷ);

# There appear to be some substantial numerical problems associated with generating samples
# jointly from all three processes, thus only f′ and noise′ are considered.
Nplot, S = 500, 10;
Xplot = linspace(0.0, 10.0, Nplot);
out = rand(rng, [f′, noise′], [Xplot, Xplot], S);

# Get posterior mean and marginals f′ and y′.




# Write results to file and run from Julia-0.6 because 0.7 can't plot anything yet...
open("vanilla_noisy_regression/posterior.csv", "w") do io
    writedlm(io, hcat(Xplot, out[1:Nplot, :], out[Nplot+1:end, :]), ',')
end

open("vanilla_noisy_regression/observations.csv", "w") do io
    writedlm(io, hcat(X, ŷ), ',')
end



###########################  Plot results - USE ONLY Julia-0.6!  ###########################

# Only run me from 0.6. PyPlot doesn't work on 0.7 at the minute.
using PyPlot


# Plot results from prior sampling.
data = readcsv("vanilla_noisy_regression/prior.csv");
X, fX, noiseX, yX = data[:, 1], data[:, 2], data[:, 3], data[:, 4];

plot(X, fX, "b", label="f");
plot(X, noiseX, "kx", label="noise");
plot(X, yX, "bx", label="y");
plot(X, fX + noiseX - yX, "r", label="f + noise - y");


# Plot results from posterior inference.
data_post = readcsv("vanilla_noisy_regression/posterior.csv");
obs = readcsv("vanilla_noisy_regression/observations.csv");
Xplot, f′Xplot, noise′Xplot = data_post[:, 1], data_post[:, 2:S+1], data_post[:, S+2:end];
X, y = obs[:, 1], obs[:, 2];

plot(Xplot, f′Xplot, "b", label="f′");
plot(X, y, "rx", label="obs");

