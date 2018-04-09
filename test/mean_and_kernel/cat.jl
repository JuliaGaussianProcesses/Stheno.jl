@testset "cat" begin

    # Test CatMean in finite case.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X1, X2 = randn(rng, N, D), randn(rng, N′, D)
        μ1, μ2 = FiniteMean(ConstantMean(1.0), X1), FiniteMean(ZeroMean{Float64}(), X2)
        μ = CatMean([μ1, μ2])
        @test length(μ) == length(μ1) + length(μ2)
        @test mean(μ) == vcat(mean(μ1), mean(μ2))
        @test CatMean(μ1, μ2) == μ
    end

    # Test CatCrossKernel in the finite case.
    let
        rng, N1, N2, N1′, N2′, D = MersenneTwister(123456), 5, 6, 2, 7, 8
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)
        X1′, X2′ = randn(rng, N1′, D), randn(rng, N2′, D)
        k11 = FiniteCrossKernel(EQ(), X1, X1′)
        k12 = FiniteCrossKernel(ZeroKernel{Float64}(), X1, X2′)
        k21 = FiniteCrossKernel(ZeroKernel{Float64}(), X2, X1′)
        k22 = FiniteCrossKernel(EQ(), X2, X2′)
        k = CatCrossKernel([k11 k12; k21 k22])
        @test size(k) == (N1 + N2, N1′ + N2′)
        @test size(k, 1) == size(k)[1]
        @test size(k, 2) == size(k)[2]
        @test xcov(k) == vcat(hcat(xcov(k11), xcov(k12)), hcat(xcov(k21), xcov(k22)))
    end

    # Test CatKernel in the finite case.
    let
        rng, N1, N2, D = MersenneTwister(123456), 5, 6, 2
        X1, X2 = randn(rng, N1, D), randn(rng, N2, D)

        # Construct CatKernel.
        k11, k22 = FiniteKernel(EQ(), X1), FiniteKernel(EQ(), X2)
        k12 = FiniteCrossKernel(ZeroKernel{Float64}(), X1, X2)
        ks_off = Matrix{FiniteCrossKernel}(undef, 2, 2)
        ks_off[1, 2] = k12
        k = CatKernel([k11, k22], ks_off)

        @test size(k) == (N1 + N2, N1 + N2)
        @test size(k, 1) == size(k)[1]
        @test size(k, 1) == size(k, 2)
        manual = vcat(hcat(Matrix(cov(k11)), xcov(k12)), hcat(zeros(N2, N1), Matrix(cov(k22))))
        @test cov(k) == Stheno.LazyPDMat(manual)
    end
end
