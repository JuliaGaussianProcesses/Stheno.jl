@testset "cat" begin

    # Test CatMean.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X1, X2 = randn(rng, N, D), randn(rng, N′, D)
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        μ = CatMean([μ1, μ2])

        @test μ == CatMean(μ1, μ2)
        @test mean(μ, [X1, X2]) == vcat(mean(μ1, X1), mean(μ2, X2))
    end

    # Test CatCrossKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 2, 7, 8
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)
        X1′, X2′ = randn(rng, N1′, D), randn(rng, N2′, D)
        k11, k12, k21, k22 =  EQ(), ZeroKernel{Float64}(), ZeroKernel{Float64}(), EQ()
        k = CatCrossKernel([k11 k12; k21 k22])

        @test size(k) == (Inf, Inf)
        @test size(k, 1) == Inf
        @test size(k, 2) == Inf

        row1 = hcat(xcov(k11, X1, X1′), xcov(k12, X1, X2′))
        row2 = hcat(xcov(k21, X2, X1′), xcov(k22, X2, X2′))
        @test xcov(k, [X1, X2], [X1′, X2′]) == vcat(row1, row2)
    end

    # Test CatKernel.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 3, 4, 2
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)
        X1′, X2′ = randn(rng, N1′, D), randn(rng, N2′, D)

        # Construct CatKernel.
        k11, k22, k12 = EQ(), EQ(), ZeroKernel{Float64}()
        ks_off = Matrix{Stheno.CrossKernel}(undef, 2, 2)
        ks_off[1, 2] = k12
        k = CatKernel([k11, k22], ks_off)

        @test size(k) == (Inf, Inf)
        @test size(k, 1) == Inf
        @test size(k, 2) == Inf

        row1 = hcat(Matrix(cov(k11, X1)), xcov(k12, X1, X2))
        row2 = hcat(zeros(N2, N1), Matrix(cov(k22, X2)))
        @test Matrix(cov(k, [X1, X2])) == vcat(row1, row2)

        # Compute xcov for CatKernel with infinite kernels.
        manual = vcat(hcat(xcov(k11, X1, X1′), xcov(k12, X1, X2′)),
                      hcat(xcov(k12, X2, X1′), xcov(k22, X2, X2′)),)
        @test xcov(k, [X1, X2], [X1′, X2′]) == manual
    end
end
