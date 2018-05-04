using Stheno, Plots
plotly();


###########################  Define and inspect our model  ###########################

#=
In this example we consider three functions: f1, f2, and f3, where f3(x) = f1(x) + f2(x).
=#
function model(gpc)
    f1 = GP(ConstantMean(randn()), EQ(), gpc)
    f2 = GP(EQ(), gpc)
    f3 = f1 + f2
    return f1, f2, f3
end

# Randomly sample `N1` locations at which to measure `f` using `y1`, and `N2` locations
# at which to measure `f` using `y2`.
rng, N1, N3 = MersenneTwister(123546), 10, 11;
X1, X3 = sort(rand(rng, N1) * 10), sort(rand(rng, N3) * 10);
f1, f2, f3 = model(GPC());

# Generate some toy observations of `y1` and `y2`.
ŷ1, ŷ3 = rand(rng, [f1, f3], [X1, X3]);

# Compute the posterior process.
(f1′, f2′, f3′) = (f1, f2, f3) | (f1(X1)←ŷ1, f3(X3)←ŷ3);

# Define some plotting stuff.
Nplot, S = 500, 25;
Xp = linspace(-2.5, 12.5, Nplot);

# Sample from posterior and write directly to file.
f1′Xp, f2′Xp, f3′Xp = rand(rng, [f1′, f2′, f3′], [Xp, Xp, Xp], S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μf1′, σf1′ = mean(f1′, Xp), marginal_std(f1′, Xp);
μf2′, σf2′ = mean(f2′, Xp), marginal_std(f2′, Xp);
μf3′, σf3′ = mean(f3′, Xp), sqrt.(diag(cov(f3′, Xp)));


###########################  Plot results - USE ONLY Julia-0.6!  ###########################

posterior_plot = plot();

# Posterior over `f1`.
plot!(posterior_plot, Xp, μf1′;
    linecolor=:red,
    linewidth=2.0,
    label="f1");
plot!(posterior_plot, Xp, [μf1′ μf1′];
    linewidth=0.0,
    fillrange=[μf1′ .- 3 .* σf1′, μf1′ .+ 3 * σf1′],
    fillalpha=0.3,
    fillcolor=:red,
    label="");
scatter!(posterior_plot, X1, ŷ1;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
plot!(posterior_plot, Xp, f1′Xp,
    linecolor=:red,
    linealpha=0.2,
    label="");

# Posterior over `f2`.
plot!(posterior_plot, Xp, μf2′;
    linecolor=:green,
    linewidth=2.0,
    label="f2");
plot!(posterior_plot, Xp, [μf2′ μf2′];
    linewidth=0.0,
    fillrange=[μf2′ .- 3 .* σf2′, μf2′ .+ 3 * σf2′],
    fillalpha=0.3,
    fillcolor=:green,
    label="");
plot!(posterior_plot, Xp, f2′Xp,
    linecolor=:green,
    linealpha=0.2,
    label="");

# Plot posterior over f3.
plot!(posterior_plot, Xp, μf3′;
    linecolor=:blue,
    linewidth=2.0,
    label="f3");
plot!(posterior_plot, Xp, [μf3′ μf3′];
    linewidth=0.0,
    fillrange=[μf3′ .- 3 .* σf3′, μf3′ .+ 3 * σf3′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="");
scatter!(posterior_plot, X3, ŷ3;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="");
plot!(posterior_plot, Xp, f3′Xp,
    linecolor=:blue,
    linealpha=0.2,
    label="");

display(posterior_plot);
