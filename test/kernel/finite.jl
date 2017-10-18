@testset "finite" begin
    let N = 5, rng = MersenneTwister(123456)
        x = randn(rng, N)
    end
end
