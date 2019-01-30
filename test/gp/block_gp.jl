using Random, LinearAlgebra
using Stheno: BlockGP, getblock, BlockGP, GPC, FiniteGP, BlockMean, BlockKernel,
    BlockCrossKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ
using Distributions: MvNormal, PDMat

@testset "block_gp" begin

    let
        rng, N, N′, D, gpc = MersenneTwister(123456), 3, 4, 2, GPC()
        x, x′ = randn(rng, N), randn(rng, N′)
        f = GP(randn(rng), eq(), gpc)
        fx, fx′ = FiniteGP(f, x, 1e-3), FiniteGP(f, x′, 1e-3)
        y, z = rand(rng, fx), rand(rng, fx′)

        # Construct a BlockGP over a single finite process (edge-case).
        f_single = BlockGP([f])
        fx_single = FiniteGP(f_single, BlockData([x]), 1e-3)
        fx′_single = FiniteGP(f_single, BlockData([x′]), 1e-3)

        @test mean(f_single) isa BlockMean
        @test mean(fx_single) == mean(fx)

        @test kernel(f_single) isa BlockKernel
        @test cov(fx_single) == cov(fx)

        @test kernel(f_single, f) isa BlockCrossKernel
        @test kernel(f, f_single) isa BlockCrossKernel

        @test cov(fx_single, fx) == cov(fx, fx)
        @test cov(fx_single, fx′) == cov(fx, fx′)
        @test cov(fx, fx_single) == cov(fx, fx)
        @test cov(fx′, fx_single) == cov(fx′, fx)
        @test cov(fx_single, fx′_single) == cov(fx, fx′)
        @test cov(fx′_single, fx_single) == cov(fx′, fx)
        @test cov(fx_single, fx_single) == cov(fx, fx)
        @test cov(fx′_single, fx′_single) == cov(fx′, fx′)

        @test rand(MersenneTwister(123456), fx_single) ==
            rand(MersenneTwister(123456), fx)

        # @test length(rand(rng, f_single)) == length(f_single)
        # @test size(rand(rng, f_single, 3)) == (length(f_single), 3)

        # # Construct a GP over multiple processes.
        # fs = BlockGP([f, g])
        # @test length(fs) == length(f) + length(g)
        # # @test eachindex(fs) == [eachindex(f), eachindex(g)]

        # @test mean(fs) isa BlockMean
        # @test mean(fs) == BlockMean([mean(f), mean(g)])
        # @test mean(fs) == BlockVector([mean(f), mean(g)])

        # @test getblock(Stheno.unbox(cov(fs)), 1, 1) == cov(f)
        # @test getblock(Stheno.unbox(cov(fs)), 2, 2) == cov(g)
        # @test getblock(Stheno.unbox(cov(fs)), 1, 2) == cov(f, g)
        # @test getblock(Stheno.unbox(cov(fs)), 2, 1) == cov(g, f)

        # @test cov(fs, f) isa BlockMatrix
        # @test cov(fs, g) isa BlockMatrix
        # @test cov(f, fs) isa BlockMatrix
        # @test cov(g, fs) isa BlockMatrix
        # @test size(cov(fs, f)) == (N + N′, N)
        # @test size(cov(fs, g)) == (N + N′, N′)
        # @test size(cov(f, fs)) == (N, N + N′)
        # @test size(cov(g, fs)) == (N′, N + N′)
        # @test cov(fs, f) == BlockMatrix(reshape([cov(f, f), cov(g, f)], 2, 1))

        # @test length(rand(rng, fs)) == length(fs)
        # @test size(rand(rng, fs, 3)) == (length(fs), 3)
        # @test sum(length.(rand(rng, [f, g]))) == length(f) + length(g)
        # fs, gs = rand(rng, [f, g], 4)
        # @test size(fs) == (length(f), 4) && size(gs) == (length(g), 4)

        # # Check `logpdf` for two independent processes.
        # joint, joint_obs = BlockGP([f, g]), BlockVector([y, z])
        # @test logpdf(joint, joint_obs) ≈ logpdf(f, y) + logpdf(g, z)
        # @test logpdf([f, g], [y, z]) == logpdf(joint, joint_obs)
    end
end
