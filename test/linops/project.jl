using Stheno: project, pw, Xt_A_X, Xt_A_Y, GPC, CustomMean

@testset "project" begin

    rng, N, N′, NZ, NZ′ = MersenneTwister(123456), 10, 11, 5, 3
    X, X′, Z, Z′ = randn(rng, N), randn(rng, N′), randn(rng, NZ), randn(rng, NZ′)

    # Construct projection with non-block matrices.
    f = GP(3, EQ(), GPC())
    f_pr = project(EQ(), f(Z), CustomMean(sin))

    @test mean(f_pr(X)) == pw(EQ(), X, Z) * mean(f(Z)) + sin.(X)
    @test cov(f_pr(X)) ≈ pw(EQ(), X, Z) * cov(f(Z)) * pw(EQ(), X, Z)'
    @test maximum(abs.(cov(f_pr(X), f(X′)) - pw(EQ(), X, Z) * cov(f(Z), f(X′)))) < eps()
    @test maximum(abs.(cov(f(X), f_pr(X′)) - cov(f(X), f(Z)) * pw(EQ(), Z, X′))) < eps()

    # # Construct projection with BlockGP as the thing over which we're projecting.
    # Zb = BlockData([Z, Z′])
    # ϕb = lhsfinite(BlockCrossKernel(reshape([EQ(), EQ()], :, 1)), Zb)
    # f_pr_b = project(ϕb, f(Zb), cos)
 
    # @test mean(f_pr_b(X)) == pw(EQ(), Zb, X)' * mean(f(Zb)) .+ cos.(X)
    # ϕZX, ϕZX′, ϕZ = pw(EQ(), Zb, X), pw(EQ(), Zb, X′), cov(f(Zb))
    # @test cov(f_pr_b(X)) == Xt_A_X(ϕZ, ϕZX)
    # @test cov(f_pr_b(X), f_pr_b(X′)) == Xt_A_Y(ϕZX, ϕZ, ϕZX′)
    # @test cov(f_pr_b(X), f(X′)) == ϕZX' * cov(f(Zb), f(X′))
    # @test cov(f(X), f_pr_b(X′)) == cov(f(X), f(Zb)) * ϕZX′

    # # Check that Blocked version is project is consistent with dense.
    # Zb2 = BlockData([Z[1:3], Z[4:5]])
    # ϕb2 = lhsfinite(BlockCrossKernel(reshape([EQ(), EQ()], :, 1)), Zb2)
    # f_pr_b2 = project(ϕb2, f(Zb2), sin)

    # @test mean(f_pr_b2(X)) ≈ mean(f_pr(X))
    # @test cov(f_pr_b2(X)) ≈ cov(f_pr(X))
    # @test cov(f_pr_b2(X), f_pr_b2(X′)) ≈ cov(f_pr(X), f_pr(X′))
    # @test cov(f_pr_b2(X), f(X′)) ≈ cov(f_pr(X), f(X′))
    # @test cov(f(X), f_pr_b2(X′)) ≈ cov(f(X), f_pr(X′))

end
