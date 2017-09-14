using Stheno
using Base.Test

@testset "Stheno" begin
    include("normal.jl")
    include("kernel.jl")
    include("gp.jl")
    include("covariance_matrices.jl")
end
