@testset "strided_covmat" begin

    # Test strided matrix functionality.
    let
        import Stheno.StridedPDMatrix
        rng = MersenneTwister(123456)
        A = randn(rng, 5, 5)
        K_ = Transpose(A) * A + UniformScaling(1e-6)
        K = StridedPDMatrix(chol(K_))
        x = randn(rng, 5)

        # Test invariances.
        @test maximum(abs.(full(K) - K_)) < 1e-10 # Loss of accuracy > machine-ϵ.
        @test RowVector(x) * (K_ \ x) ≈ invquad(K, x)
        @test logdet(K) ≈ logdet(K_)

        @test size(K) == size(K_)
        @test size(K, 1) == size(K_, 1)
        @test size(K, 2) == size(K_, 2)

        @test K == K
        @test chol(K) == chol(K_)
    end

    # Test covariance matrix construction with a single kernel.
    let rng = MersenneTwister(123456)
        P, Q = 5, 6
        x, y = randn(rng, 5), randn(rng, 6)
        k = Finite(EQ(), x, y)

        K = Matrix{Float64}(uninitialized, P, Q)
        @test cov!(K, k) == EQ().(x, y')
        @test cov!(K, k) == cov(k)

        k = Finite(RQ(1.0), x)
        K = Matrix{Float64}(uninitialized, P, P)
        @test cov!(K, k) == RQ(1.0).(x, x')
        @test cov!(K, k) == cov(k)
    end

    # Test covariance matrix construct with multiple kernels.
    let rng = MersenneTwister(123456)
        P1, P2, Q1, Q2 = 3, 4, 5, 6
        x1, x2, y1, y2 = randn.(rng, [P1, P2, Q1, Q2])
        k11, k12, k21, k22 = Finite.(
            [EQ(), RQ(1.0), Stheno.Constant(2.5), Linear(4.0)],
            [x1, x1, x2, x2],
            [y1, y2, y1, y2],
        )

        K = Matrix{Float64}(uninitialized, P1 + P2, Q1 + Q2)
        ks = [k11 k12; k21 k22]
        @test cov!(K, ks) == vcat(hcat(cov(k11), cov(k12)), hcat(cov(k21), cov(k22)))
        @test cov!(K, ks) == cov(ks)
    end
end
