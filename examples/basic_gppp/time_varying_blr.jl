# Please run from the `basic_gppp` directory.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Stheno, Plots, Random, ColorTypes, FixedPointNumbers
using Stheno: @model

###########################  Define and inspect our model  ###########################

#=
g1 and g2 are time-varying basis functions. In a real application, these might be specified
by data.

w1 and w2 are the time-varying basis functions for these bases, specified by slowly
varying GPs.

f is the prediction of the regressor.

y is the addition of f and rough temporally-correlated "noise".
=#
@model function model()
    g1, g2 = x->x / 4, cos
    w1, w2 = stretch(GP(EQ()), 0.2), stretch(GP(EQ()), 1)
    f = g1 * w1 + g2 * w2
    y = f + 0.3 * GP(Matern12())
    return w1, w2, f, y
end

# Sample from the prior from plotting and for conditioning.
rng, N, Nplot, S = MersenneTwister(123456), 250, 500, 100;
X, Xp = sort(rand(rng, N) * 10), range(-2.5, stop=12.5, length=Nplot);
w1, w2, f, y = model();
w1s, w2s, fs, ŷ = rand(rng, [w1(Xp, 1e-9), w2(Xp, 1e-9), f(Xp, 1e-9), y(X, 1e-9)]);

# Compute posterior distribution over f′.
w1′, w2′, f′, y′ = (w1, w2, f, y) | (y(X) ← ŷ);

# Sample from the posterior and write to file.
w1′s, w2′s, y′s = rand(rng, [w1′(Xp, 1e-9), w2′(Xp, 1e-9), y′(Xp, 1e-9)], S);

# Get posterior mean and marginals f′ and y′ and write them for plotting.

ms_w1′ = marginals(w1′(Xp));
ms_w2′ = marginals(w2′(Xp));
ms_y′ = marginals(y′(Xp));

μw1′, σw1′ = mean.(ms_w1′), std.(ms_w1′);
μw2′, σw2′ = mean.(ms_w2′), std.(ms_w2′);
μy′, σy′ = mean.(ms_y′), std.(ms_y′);



###########################  Plot results ###########################

gr();
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


# Plot posterior over w1.
plot!(posterior_plot, Xp, μw1′;
    linecolor=:green,
    linewidth=2.0,
    label="w1",
);
plot!(posterior_plot, Xp, [μw1′ μw1′];
    linewidth=0.0,
    fillrange=[μw1′ .- 3 .* σw1′, μw1′ .+ 3 * σw1′],
    fillalpha=0.2,
    fillcolor=:green,
    label="",
);
plot!(posterior_plot, Xp, w1′s;
    linecolor=:green,
    linealpha=0.1,
    label="",
);

# Plot posterior over w2.
plot!(posterior_plot, Xp, μw2′;
    linecolor=:magenta,
    linewidth=2.0,
    label="w2",
);
plot!(posterior_plot, Xp, [μw2′ μw2′];
    linewidth=0.0,
    fillrange=[μw2′ .- 3 .* σw2′, μw2′ .+ 3 * σw2′],
    fillalpha=0.2,
    fillcolor=:magenta,
    label="",
);
plot!(posterior_plot, Xp, w2′s;
    linecolor=:magenta,
    linealpha=0.1,
    label="",
);


# Plot x1 and x2
plot!(posterior_plot, Xp, (x->x / 4).(Xp);
    linecolor=:black,
    linewidth=1.0,
    label="sin",
);
plot!(posterior_plot, Xp, cos.(Xp);
    linecolor=:black,
    linewidth=1.0,
    linestyle=:dash,
    label="cos",
);

# Plot samples against which we're regressing.
scatter!(posterior_plot, X, ŷ;
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

# Plot posterior over `y`.
plot!(posterior_plot, Xp, μy′;
    linecolor=:blue,
    linewidth=2.0,
    label="y",
);
plot!(posterior_plot, Xp, [μy′ μy′];
    linewidth=0.0,
    fillrange=[μy′ .- 3 .* σy′, μy′ .+ 3 * σy′],
    fillalpha=0.3,
    fillcolor=:blue,
    label="",
);

# savefig(posterior_plot, "plotting_research_talk/tv_blr.pdf");

display(posterior_plot);
