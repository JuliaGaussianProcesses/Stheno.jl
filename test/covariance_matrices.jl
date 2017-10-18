@testset "strided_covmat" begin

    # Test strided matrix functionality.
    let
        import Stheno.StridedPDMatrix
        rng = MersenneTwister(123456)
        A = randn(rng, 5, 5)
        K_ = A.'A + UniformScaling(1e-6)
        K = StridedPDMatrix(chol(K_))
        x = randn(rng, 5)

        # Test invariances.
        @test maximum(abs.(full(K) - K_)) < 1e-10 # Loss of accuracy > machine-ϵ.
        @test x.' * (K_ \ x) ≈ invquad(K, x)
        @test logdet(K) ≈ logdet(K_)

        @test size(K) == size(K_)
        @test size(K, 1) == size(K_, 1)
        @test size(K, 2) == size(K_, 2)

        @test K == K
        @test chol(K) == chol(K_)
    end

    # Test covariance matrix construction.
    let
        import Stheno.Constant
        rng = MersenneTwister(123456)
        k = RQ(1.0)
        P, Q = 5, 7
        x, y = randn(rng, P), randn(rng, Q)
        @test cov(k, x) isa StridedPDMatrix
        @test cov(k, x) isa AbstractPDMat
        @test full(cov(k, x)) ≈ k.(x, RowVector(x))
        @test full(cov(k, RowVector(x))) == full(cov(k, x))

        @test cov(k, x, y) == k.(x, RowVector(y))
        @test cov(k, y, x) == cov(k, x, y).'
        @test cov(k, RowVector(x), RowVector(y)) == cov(k, x, y)
    end

    # Test joint covariance matrix construction.
    let
        rng = MersenneTwister(123456)
        P, Q = 3, 2
        x, y = randn(rng, P), randn(rng, Q)

        gpc = GPCollection()
        gp1, gp2 = GP(gpc, x->0.0, EQ()), GP(gpc, x->0.0, RQ(1.0))
        K1, K2 = full(cov(kernel(gp1), x)), full(cov(kernel(gp2), y))
        @test K1 == full(cov([(gp1, x)]))
        @test K2 == full(cov([(gp2, y)]))

        K_manual = vcat(hcat(K1, zeros(P, Q)), hcat(zeros(Q, P), K2))
        @test K_manual == full(cov([(gp1, x), (gp2, y)]))
    end

    # Test joint cross-covariance matrix construction.
    let
        rng = MersenneTwister(123456)
        P1, P2, Ps1, Ps2 = 3, 2, 4, 5
        x1, x2, xs1, xs2 = randn.(rng, [P1, P2, Ps1, Ps2])

        gpc = GPCollection()
        gp1, gp2 = GP(gpc, x->0.0, EQ()), GP(gpc, x->0.0, RQ(1.0))

        K11, K12 = full(cov(kernel(gp1, gp1), xs1, x1)), full(cov(kernel(gp1, gp2), xs1, x2))
        K21, K22 = full(cov(kernel(gp2, gp1), xs2, x1)), full(cov(kernel(gp2, gp2), xs2, x2))

        K_manual = vcat(hcat(K11, K12), hcat(K21, K22))
        @test K_manual == full(cov([(gp1, xs1), (gp2, xs2)], [(gp1, x1), (gp2, x2)]))
    end
end
