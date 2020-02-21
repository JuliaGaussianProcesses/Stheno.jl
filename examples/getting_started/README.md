# Examples: Getting Started

Here we provide a terse but reasonably comprehensive introduction to the API made available in Stheno. This is a helpful introduction if you're looking to build on Stheno in your own code. Where appropriate, all operations that should be algorithmically differentiable.

`basic_operations.jl` illustrates the basic manipulations of Gaussian processes available in this package. 

`high_level_plotting.jl` shows how to effectively combine `Plots.jl` with this package to quickly generate plots for 1D input spaces, while `low_level_plotting.jl` provides an example of the same plotting functionality, but without any of the high-level plotting utilities provided by this package -- this is important if the high-level plotting utilities aren't suitable for your application.
