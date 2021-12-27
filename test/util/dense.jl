@timedtestset "dense" begin
    @timedtestset "fdm stuff" begin
        rng, Ps, Qs = MersenneTwister(123456), [5, 4], [3, 2, 1]
        X = mortar([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
        vec_X, from_vec = FiniteDifferences.to_vec(X)
        @test vec_X isa Vector
        @test from_vec(vec_X) == X
    end
    @timedtestset "Stheno._collect ∘ mortar" begin
        @timedtestset "BlockVector" begin

            # Generate some blocks.
            Ps = [5, 6, 7]
            x = BlockArray(randn(sum(Ps)), Ps).blocks

            # Verify the pullback.
            ȳ = randn(sum(Ps))
            adjoint_test(Stheno._collect ∘ mortar, ȳ, x)
        end
        @timedtestset "BlockMatrix" begin

            # Generate some blocks.
            Ps = [3, 4, 5]
            Qs = [6, 7, 8, 9]
            X = BlockArray(randn(sum(Ps), sum(Qs)), Ps, Qs).blocks
            Ȳ = randn(sum(Ps), sum(Qs))

            # Verify pullback.
            adjoint_test(Stheno._collect ∘ mortar, Ȳ, X)
        end
    end
end
