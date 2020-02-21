#
# Here we demonstrate how to use Stheno's built in plotting functionality to make
# visualising 1D examples straightforward.
#

# Load some other code and construct some basic objects. Please see the file for reference.
include("basic_operations.jl")

# Specify our plotting backend.
gr();

# Construct a new plot object.
posterior_plot = plot();

# Plot the posterior distribution.
plot!(posterior_plot, f_post(xpr); samples=5, color=:blue);

# Plot the observations.
scatter!(posterior_plot, x, y;
    markercolor=:blue,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);

# Show the plot.
display(posterior_plot);
