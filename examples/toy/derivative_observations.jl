# This example is a WIP. It works, but is a bit rough around the edges.

using Stheno, Random, Distributions, Plots, ColorTypes, FixedPointNumbers
using Stheno: @model, ∇

rng = MersenneTwister(12345);

@model function gp()
    f = GP(EQ())
    ∇f = ∇(f)
    return f, ∇f
end

f, ∇f = gp()

# Sample prior at random locations.
Xf = rand(Uniform(-5, 5), 5)
X∇f = rand(Uniform(-5, 5), 15)
X∇f = Xf
f_, ∇f_ = rand([f(Xf), ∇f(X∇f)])

# Compute log prob. of samples.
logpdf([f(Xf), ∇f(X∇f)], [f_, ∇f_])

# Compute posterior processes.
f′, ∇f′ = (f, ∇f) | (f(Xf) ← f_, ∇f(X∇f) ← ∇f_)



#######################  Do posterior inference give a few samples  #######################

Np, S = 500, 10;
Xp = range(-6; stop=6, length=Np);
f′Xp = rand(rng, f′(Xp), S);

∇Xp = range(-6; stop=6, length=Np);
∇f′Xp = rand(rng, ∇f′(∇Xp), S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.
μ′f, σ′f = marginals(f′(Xp));
μ′∇f, σ′∇f = marginals(∇f′(∇Xp));



####################################  Plot results  ####################################

plotly();

posterior_plot = plot(
    legend=:topright,
    legendfont=Plots.Font(
        "sans-serif",
        10,
        :hcenter,
        :vcenter,
        0.0,
        RGB{Normed{UInt8, 8}}(0.0,0.0,0.0)
    ),
    background_color_legend=RGBA(1, 1, 1, 0),
    foreground_color_legend=RGBA(1, 1, 1, 0),
);

plot!(posterior_plot, Xp, μ′f;
    linewidth=2.0,
    linecolor=:blue,
    label="");
plot!(posterior_plot, Xp, μ′f;
    linewidth=0.0,
    linecolor=:blue,
    linealpha=0.2,
    fillrange=[μ′f.- 3  .* σ′f],
    fillalpha=0.2,
    fillcolor=:blue,
    label=["posterior, f", ""]);
plot!(posterior_plot, Xp, f′Xp[:, 1];
    linecolor=:blue,
    linealpha=0.3,
    label="posterior samples, f");
plot!(posterior_plot, Xp, μ′f;
    linewidth=0.0,
    linecolor=:blue,
    linealpha=0.2,
    fillrange=[μ′f .+ 3 .* σ′f],
    fillalpha=0.2,
    fillcolor=:blue,
    label="");
plot!(posterior_plot, Xp, f′Xp;
    linecolor=:blue,
    linealpha=0.3,
    label="");
scatter!(posterior_plot, Xf, f_;
    markercolor=:green,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=5,
    markeralpha=0.9,
    label="f_");

colour = :red
plot!(posterior_plot, ∇Xp, μ′∇f;
    linewidth=2.0,
    linecolor=colour,
    label="");
plot!(posterior_plot, ∇Xp, μ′∇f;
    linewidth=0.0,
    linecolor=colour,
    linealpha=0.2,
    fillrange=[μ′∇f.- 3  .* σ′∇f],
    fillalpha=0.2,
    fillcolor=colour,
    label=["posterior, ∇f", ""]);
plot!(posterior_plot, ∇Xp, ∇f′Xp[:, 1];
    linecolor=colour,
    linealpha=0.3,
    label="posterior samples, ∇f");
plot!(posterior_plot, ∇Xp, μ′∇f;
    linewidth=0.0,
    linecolor=colour,
    linealpha=0.2,
    fillrange=[μ′∇f .+ 3 .* σ′∇f],
    fillalpha=0.2,
    fillcolor=colour,
    label="");
plot!(posterior_plot, ∇Xp, ∇f′Xp;
    linecolor=colour,
    linealpha=0.3,
    label="");
scatter!(posterior_plot, X∇f, ∇f_;
    markercolor=colour,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=3,
    markeralpha=0.9,
    label="∇f_");

display(posterior_plot);
