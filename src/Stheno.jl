module Stheno

using PDMats, Distributions

# Kernel and covariance 
include("kernel.jl")
include("strided_covmat.jl")

# Multivariate Normal and GP.
include("normal.jl")

end # module
