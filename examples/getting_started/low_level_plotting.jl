#
# Here we demonstrate how to build up a plot manually using just the Plots.jl interface and
# the quantities that are easily computable using Stheno. This approach involves a
# reasonable amount of boilerplate, but demonstates how to work at a lower level in case the
# high-level plotting functionality demonstrated in high_level_plotting.jl is insufficient
# for your use case.
#

# Load some other code and construct some basic objects. Please see the file for reference.
include("basic_operations.jl")

# Specify our plotting backend.
gr();

# Construct a new plot object.
posterior_plot = plot(legend=nothing);

# Generate several samples from the posterior predictive distribution.
Y = rand(rng, f_post(xpr), 5)
m = mean.(posterior_marginals)
σ = std.(posterior_marginals)

plot!(posterior_plot, xpr, m;
    linecolor=:blue,
    linewidth=2,
);
plot!(posterior_plot, xpr, [m m];
    linewidth=0.0,
    linecolor=:blue,
    fillrange=[m .- 3 .* σ, m .+ 3 * σ],
    fillalpha=0.3,
    fillcolor=:blue,
);
plot!(posterior_plot, xpr, Y;
    linewidth=0.5,
    linealpha=0.5,
    linecolor=:blue,
);

# Plot the observations.
scatter!(posterior_plot, x, y;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
);

# Show the plot.
display(posterior_plot);
