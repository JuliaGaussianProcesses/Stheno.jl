using Stheno, Plots, Optim, Distributions

###########################  Define and inspect our model  ###########################

# Define the vanilla GP-regression generative model.
function model(θ::Vector{<:Real})

    gpc = GPC()

    # Define a smooth latent process.
    f = GP(ITKernel(EQ(), x->exp(θ[1]) * x), gpc)

    # Define a latent noise process.
    noise = GP(Noise(exp(θ[2])), gpc)

    # Sum them to get the process of which we shall make observations.
    y = f + noise

    # Return all three processes so that we can inspect all of them.
    return f, noise, y
end

# Get the third output from model.
model_obs(θ) = model(θ)[3]

# Generate some input locations and sample from the prior.
rng, N = MersenneTwister(123546), 500;
X_prior = sort(rand(rng, N) * 10);
f, noise, y = model(log.([1.0 , 1e-2]));
fX, noiseX, yX = rand(rng, [f, noise, y], [X_prior, X_prior, X_prior]);


###############  Do learning and posterior inference give a few samples  #################

# Sample a few points from the prior and compute the posterior processes.
N = 25;
X = sort(rand(rng, N) * 10);
ŷ = rand(rng, y, X);

# Optimise the model parameters using a gradient-free method.
options = Optim.Options(show_trace=true, iterations=100);
θ0 = zeros(2);
θ0 = log.([0.01, 1e-3])

function logpdf_prior(θ)
    logpdf_l = logpdf(InverseGamma(0.05, 0.1), exp(θ[1]))
    logpdf_σ² = logpdf(InverseGamma(0.05, 0.1), exp(θ[2]))
    return logpdf_l + logpdf_σ²
end
opt = optimize(θ->-logpdf(model_obs(θ)(X), ŷ) - logpdf_prior(θ), θ0, NelderMead(), options);
θ_opt = opt.minimizer;


# Get the model using the learned parameters.
f, noise, y = model(θ_opt);
f′, noise′, y′ = (f, noise, y) | (y(X) ← ŷ);

# Sample jointly from the posterior.
Nplot, S = 500, 100;
Xplot = linspace(-2.0, 12.0, Nplot);
f′Xp, noise′Xp = rand(rng, [f′, noise′], [Xplot, Xplot], S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf′, σf′ = marginals(f′, Xplot);
σy′ = diag_std(y′, Xplot);



####################################  Plot results  ####################################

plotly();

prior_plot = plot(X_prior, fX, label="f");
scatter!(prior_plot, X_prior, noiseX,
    label="noise",
    markershape=:xcross,
    markerstrokewidth=0.0,
    markersize=2);
scatter!(prior_plot, X_prior, yX,
    label="y",
    markershape=:xcross,
    markerstrokewidth=0.0,
    markersize=2);
plot!(prior_plot, X_prior, fX + noiseX - yX,
    label="f + noise - y",
    linecolor=:red);

posterior_plot = plot(Xplot, f′Xp;
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
scatter!(posterior_plot, X, ŷ;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    legend=false);
joint_plot = plot(prior_plot, posterior_plot, layout=(2, 1));
display(joint_plot);
