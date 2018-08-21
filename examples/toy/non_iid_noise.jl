using Stheno, Plots, Random
using Stheno: @model

###########################  Define and inspect our model  ###########################

#=
We wish to perform inference in some latent process `f`, but it is corrupted by both some
iid noise, and some non-iid noise. Specifically, ω is some periodic behaviour in which we
are uninterested. Similarly, g is also drawn from a GP with an EQ kernel, but with a much
shorter length-scale. We wish to use observations of
`f + ω + g + GP(ZeroMean{Float64}(), Noise(0.1))`
to infer `f`. In this context, `ω + g + GP(ZeroMean{Float64}(), Noise(0.1))` is
regarded as our "noise process".

The described model is unidentiable. Conditioning on an observation of the periodic process
at some point, however, allows us to infer a particular `f`. Use the `model_identifiable`
function rather than `model` to see the results of this. Note the example of sequential
conditioning.
=#
@model function model()
    f = GP(EQ())
    ω = GP(periodic(EQ(), 1))
    return f, ω, ω + f, ω + f + GP(Noise(0.1))
end
@model function model_identifiable()
    f = GP(EQ())
    t = GP(periodic(EQ(), 1))
    ω = t | (t([0])←[0.0])
    return f, ω, ω + f, ω + f + GP(Noise(0.001))
end

# Select some input locations and sample from the prior.
rng, N, Nplot, S = MersenneTwister(123456), 100, 500, 100;
X, Xp = sort(rand(rng, N) * 10), range(-2.5, stop=12.5, length=Nplot);
f, ω, fpω, y = model();
fs, ωs, ys, ŷ = rand(rng, [f(Xp), ω(Xp), fpω(Xp), y(X)]);

# Compute posterior distribution over f′.
f′ = f | (y(X) ← ŷ);

# Sample from the posterior and write to file.
f′Xp = rand(rng, f′(Xp), S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf′, σf′ = marginals(f′(Xp));



###########################  Plot results - USE ONLY Julia-0.6!  ###########################

posterior_plot = plot();

# Prior over `f`.
plot!(posterior_plot, Xp, ys;
    linewidth=1.0,
    linecolor=:red,
    label="y");
plot!(posterior_plot, Xp, fs;
    linewidth=2.0,
    linecolor=:black,
    label="f");

# Posterior over `f`.
plot!(posterior_plot, Xp, μf′;
    linecolor=:blue,
    linewidth=2.0,
    label="f1");
plot!(posterior_plot, Xp, [μf′ μf′];
    linewidth=0.0,
    fillrange=[μf′ .- 3 .* σf′, μf′ .+ 3 * σf′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");
scatter!(posterior_plot, X, ŷ;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
plot!(posterior_plot, Xp, f′Xp,
    linecolor=:blue,
    linealpha=0.2,
    label="");

display(posterior_plot);
