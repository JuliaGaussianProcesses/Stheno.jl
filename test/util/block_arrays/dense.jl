@timedtestset "dense" begin

    @testset "Array ∘ mortar" begin
        @testset "BlockVector" begin

            # Generate some blocks.
            Ps = [5, 6, 7]
            x = BlockArray(randn(sum(Ps)), Ps).blocks

            # Verify the pullback.
            ȳ = randn(sum(Ps))
            adjoint_test(Array ∘ mortar, ȳ, x)
        end
        @testset "BlockMatrix" begin

            # Generate some blocks.
            Ps = [3, 4, 5]
            Qs = [6, 7, 8, 9]
            X = BlockArray(randn(sum(Ps), sum(Qs)), Ps, Qs).blocks
            Ȳ = randn(sum(Ps), sum(Qs))

            # Verify pullback.
            adjoint_test(Array ∘ mortar, Ȳ, X)
        end
    end
end
