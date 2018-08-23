using Stheno, Plots, Distributions, ColorTypes, Random, FixedPointNumbers
using Stheno: @model, Titsias


rng = MersenneTwister(12345);

###########################  Define and inspect our model  ###########################

# Specify generative model.
@model function gp(σ²)
    f = 1.5 * GP(EQ())
    y = f + GP(Noise(σ²))
    return f, y
end

# Construct generative model.
σ² = 1.0;
f, y = gp(σ²);

# Sample prior at random locations.
Xy = rand(Uniform(-5, 5), 100);
y_ = rand(y(Xy));

# Approximately condition processes. Play around with different lengths of Z.
Z = range(-5, stop=5, length=10);
f′ = f | Titsias(f(Xy)←y_, f(Z), sqrt(σ²));
f′_exact = f | (y(Xy)←y_);



#######################  Do posterior inference give a few samples  #######################

Np, S = 500, 10;
Xp = range(-6; stop=6, length=Np);
f′Xp = rand(rng, f′(Xp), S);

# Get posterior marginals for approximate and exact conditioning.
μ′f, σ′f = marginals(f′(Xp));
μf′_exact, σf′_exact = marginals(f′_exact(Xp));



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

plot!(posterior_plot, Xp, μf′_exact;
    linewidth=2.0,
    linecolor=:blue,
    label="");
plot!(posterior_plot, Xp, μf′_exact;
    linewidth=0.0,
    linecolor=:blue,
    linealpha=0.2,
    fillrange=[μf′_exact.- 3  .* σf′_exact],
    fillalpha=0.2,
    fillcolor=:blue,
    label=["exact posterior, f", ""]);
plot!(posterior_plot, Xp, μf′_exact;
    linewidth=0.0,
    linecolor=:blue,
    linealpha=0.2,
    fillrange=[μf′_exact .+ 3 .* σf′_exact],
    fillalpha=0.2,
    fillcolor=:blue,
    label="");

plot!(posterior_plot, Xp, μ′f;
    linewidth=2.0,
    linecolor=:red,
    label="");
plot!(posterior_plot, Xp, μ′f;
    linewidth=0.0,
    linecolor=:red,
    linealpha=0.2,
    fillrange=[μ′f.- 3  .* σ′f],
    fillalpha=0.2,
    fillcolor=:red,
    label=["posterior, f", ""]);
plot!(posterior_plot, Xp, μ′f;
    linewidth=0.0,
    linecolor=:red,
    linealpha=0.2,
    fillrange=[μ′f .+ 3 .* σ′f],
    fillalpha=0.2,
    fillcolor=:red,
    label="");

scatter!(posterior_plot, Xy, y_;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=6,
    markeralpha=0.3,
    label="y_");

display(posterior_plot);
