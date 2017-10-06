module Stheno

using PDMats

include("normal.jl")

# All covariance matrix functionality.
include("kernel/base.jl")
include("kernel/compose.jl")
include("kernel/transform.jl")
include("kernel/input_transform.jl")

include("gp.jl")
include("covariance_matrices.jl")

end # module
