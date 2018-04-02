@testset "posterior" begin

    import Stheno: LhsFinite, ConditionalData, Conditional, Constant

    # Test that ConditionalData works as expected.
    let rng = MersenneTwister(123456)
        x = randn(rng, 10, 10)
        U = chol(Transpose(x) * x + 1e-9I)
        data = Stheno.ConditionalData(U)
        @test data.U == U
        @test length(data.tmp) == 10
        @test length(data.tmp′) == 10
    end

    # A couple of posterior checks.
    let rng = MersenneTwister(123456), N = 5

        x = randn(rng, N)
        k1, k2, k12 = EQ(), RQ(1.0), Constant(0.0)
        data, k1f̂ = ConditionalData(chol(k1.(x, x') + 1e-9I)), LhsFinite(k1, x)

        kpost_1 = Conditional(k1, Vector{Kernel}([k1f̂]), Vector{Kernel}([k1f̂]), data)
        @test all(abs.(kpost_1.(x, x')) .< 1e-8)

        k2f̂ = LhsFinite(k12, x)
        kpost_21 = Conditional(k2, Vector{Kernel}([k2f̂]), Vector{Kernel}([k2f̂]), data)
        @test all(abs.(kpost_21.(x, x') .- RQ(1.0).(x, x')) .< 1e-12)
    end

    # Test conditioning on multiple input kernels.
    let rng = MersenneTwister(123456)

        x, y, g = randn(rng, 3), randn(rng, 4), randn(rng, 10)
        k1, k2, kx = EQ(), EQ(), EQ()

        K = cov([Finite(k1, x) Finite(kx, x, y); Finite(kx, y, x) Finite(k2, y)])
        data, k1f̂, k2f̂ = ConditionalData(chol(K)), LhsFinite(k1, x), LhsFinite(k2, y)

        # Check that posterior is close to zero at observations.
        k_post = Conditional(k1, Vector{Kernel}([k1f̂, k2f̂]), Vector{Kernel}([k1f̂, k2f̂]), data)
        @test all(abs.(k_post.(x, x') .< 1e-8))
        @test all(abs.(k_post.(y, y') .< 1e-8))
        @test all(abs.(k_post.(x, y') .< 1e-8))
        @test all(abs.(k_post.(y, x') .< 1e-8))

        # Check that the results agree with the corresponding single kernel.
        z = vcat(x, y)
        data, kf̂ = ConditionalData(chol(cov(Finite(k1, z)))), LhsFinite(k1, z)
        k_post_single = Conditional(k1, Vector{Kernel}([kf̂]), Vector{Kernel}([kf̂]), data)
        @test k_post.(x, x') == k_post_single.(x, x')
        @test k_post.(x, y') == k_post_single.(x, y')
        @test k_post.(y, x') == k_post_single.(y, x')
        @test k_post.(y, y') == k_post_single.(y, y')
        @test k_post.(g, g') == k_post_single.(g, g')
    end
end
