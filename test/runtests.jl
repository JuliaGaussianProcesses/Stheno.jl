include("front_matter.jl")
@testset "Stheno" begin
    include("input_collection_types.jl")

    println("gp:")
    @testset "gp" begin
        include(joinpath("gp", "util.jl"))
        include(joinpath("gp", "atomic_gp.jl"))
        include(joinpath("gp", "derived_gp.jl"))
        include(joinpath("gp", "sparse_finite_gp.jl"))
    end

    println("affine_transformations:")
    @testset "affine_transformations" begin
        include(joinpath("affine_transformations", "cross.jl"))
        include(joinpath("affine_transformations", "addition.jl"))
        include(joinpath("affine_transformations", "compose.jl"))
        include(joinpath("affine_transformations", "product.jl"))
        include(joinpath("affine_transformations", "additive_gp.jl"))
    end

    include("gaussian_process_probabilistic_programme.jl")
end
