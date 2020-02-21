# Examples: Basic GPPP

Various simply toy GPPP examples.

- `process_decomposition.jl`: additive GP where we inspect the posterior over the processes
- `sensor_fusion.jl`: integrate multiple types of observations of a process made under different types of noise.
- `time_varying_blr.jl`: Bayesian Linear Regression (BLR) in which the coefficients vary throughout the input space, each according to an independent Gaussian process. Here it's 1D, so we call it time.
