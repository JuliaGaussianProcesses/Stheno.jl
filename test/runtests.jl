using Stheno
using Base.Test

@testset "Stheno" begin
    include("kernel.jl")
    include("strided_covmat.jl")
    include("normal.jl")
end
