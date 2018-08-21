using Stheno, Plots, Random
using Stheno: @model

###########################  Define and inspect our model  ###########################

#=

=#
@model function model()
    g1, g2 = x::Real->sin(x), x::Real->x
    w1, w2 = GP(EQ()), GP(EQ())
    f = g1 * w1 + g2 * w2
    y = f + GP(Noise(0.001))
    return w1, w2, f, y
end

# Sample from the prior from plotting and for conditioning.
rng, N, Nplot, S = MersenneTwister(123456), 250, 500, 100;
X, Xp = sort(rand(rng, N) * 10), range(-2.5, stop=12.5, length=Nplot);
w1, w2, f, y = model();
w1s, w2s, fs, ŷ = rand(rng, [w1(Xp), w2(Xp), f(Xp), y(X)]);

# Compute posterior distribution over f′.
w1′, w2′, f′, y′ = (w1, w2, f, y) | (y(X) ← ŷ);

# Sample from the posterior and write to file.
# w1′s, w2′s, f′s = rand(rng, [w1′(Xp), w2′(Xp), f′(Xp)], S);
w1′s, w2′s, y′s = rand(rng, [w1′(Xp), w2′(Xp), y′(Xp)], S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μw1′, σw1′ = marginals(w1′(Xp));
μw2′, σw2′ = marginals(w2′(Xp));
# μf′, σf′ = marginals(f′(Xp));
μy′, σy′ = marginals(y′(Xp));

###########################  Plot results - USE ONLY Julia-0.6!  ###########################

posterior_plot = plot();

# Plot samples against which we're regressing.
plot!(posterior_plot, X, ŷ;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");

# # Plot posterior over `f`.
# plot!(posterior_plot, Xp, μf′;
#     linecolor=:blue,
#     linewidth=2.0,
#     label="f");
# plot!(posterior_plot, Xp, [μf′ μf′];
#     linewidth=0.0,
#     fillrange=[μf′ .- 3 .* σf′, μf′ .+ 3 * σf′],
#     fillalpha=0.3,
#     fillcolor=:blue,
#     label="");
# plot!(posterior_plot, Xp, f′s;
#     linecolor=:blue,
#     linealpha=0.2,
#     label="");

# Plot posterior over `y`.
plot!(posterior_plot, Xp, μy′;
    linecolor=:blue,
    linewidth=2.0,
    label="y");
plot!(posterior_plot, Xp, [μy′ μy′];
    linewidth=0.0,
    fillrange=[μy′ .- 3 .* σy′, μy′ .+ 3 * σy′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");
plot!(posterior_plot, Xp, y′s;
    linecolor=:blue,
    linealpha=0.2,
    label="");


# Plot posterior over w1.
plot!(posterior_plot, Xp, μw1′;
    linecolor=:green,
    linewidth=2.0,
    label="w1");
plot!(posterior_plot, Xp, [μw1′ μw1′];
    linewidth=0.0,
    fillrange=[μw1′ .- 3 .* σw1′, μw1′ .+ 3 * σw1′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
plot!(posterior_plot, Xp, w1′s;
    linecolor=:green,
    linealpha=0.2,
    label="");


# Plot posterior over w2.
plot!(posterior_plot, Xp, μw2′;
    linecolor=:magenta,
    linewidth=2.0,
    label="w2");
plot!(posterior_plot, Xp, [μw2′ μw2′];
    linewidth=0.0,
    fillrange=[μw2′ .- 3 .* σw2′, μw2′ .+ 3 * σw2′],
    fillalpha=0.3,
    fillcolor=:magenta,
    label="");
plot!(posterior_plot, Xp, w2′s;
    linecolor=:magenta,
    linealpha=0.2,
    label="");

display(posterior_plot);
