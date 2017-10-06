using Stheno
using Base.Test

@testset "Stheno" begin

    include("normal.jl")

    @testset "Kernel" begin
        include("kernel/base.jl")
        include("kernel/compose.jl")
        include("kernel/transform.jl")
        include("kernel/input_transform.jl")
    end

    include("gp.jl")
    include("covariance_matrices.jl")

end
