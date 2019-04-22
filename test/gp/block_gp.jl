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

        @test mean(fx_single) == mean(fx)

        @test cov(fx_single) == cov(fx)

        @test cov(fx_single, fx) == cov(fx, fx)
        @test cov(fx_single, fx′) == cov(fx, fx′)
        @test cov(fx, fx_single) == cov(fx, fx)
        @test cov(fx′, fx_single) == cov(fx′, fx)
        @test cov(fx_single, fx′_single) == cov(fx, fx′)
        @test cov(fx′_single, fx_single) == cov(fx′, fx)
        @test cov(fx_single, fx_single) == cov(fx, fx)
        @test cov(fx′_single, fx′_single) == cov(fx′, fx′)

        y = rand(MersenneTwister(123456), fx_single)
        ŷ = rand(MersenneTwister(123456), fx)
        @show typeof(y), typeof(ŷ)
        @test y == ŷ
        @test logpdf(fx, y) == logpdf(fx_single, y)


        # Construct a GP over multiple processes.
        fs = BlockGP([f, f])
        fs_x = FiniteGP(fs, BlockData([x, x′]), 1e-3)

        @test mean(fs_x) == vcat(mean(fx), mean(fx′))

        @test cov(fs_x)[1:N, 1:N] == cov(fx)
        @test cov(fs_x)[1:N, N+1:end] == cov(fx, fx′)
        @test cov(fs_x)[N+1:end, 1:N] == cov(fx′, fx)
        @test cov(fs_x)[N+1:end, N+1:end] == cov(fx′)

        @test cov(fs_x, fs_x)[1:N, 1:N] == cov(fx, fx)
        @test cov(fs_x, fs_x)[1:N, N+1:end] == cov(fx, fx′)
        @test cov(fs_x, fs_x)[N+1:end, 1:N] == cov(fx′, fx)
        @test cov(fs_x, fs_x)[N+1:end, N+1:end] == cov(fx′, fx′)

        @test cov(fs_x, fx) == vcat(cov(fx, fx), cov(fx′, fx))
        @test cov(fs_x, fx′) == vcat(cov(fx, fx′), cov(fx′, fx′))
        @test cov(fx, fs_x) == hcat(cov(fx, fx), cov(fx, fx′))
        @test cov(fx′, fs_x) == hcat(cov(fx′, fx), cov(fx′, fx′))
    end
end
