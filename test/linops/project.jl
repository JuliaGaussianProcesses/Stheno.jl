using Stheno: project, pairwise, lhsfinite, unbox, Xt_A_X, Xt_A_Y

@testset "project" begin

let
    rng, N, N′, NZ, NZ′ = MersenneTwister(123456), 10, 11, 5, 3
    X, X′, Z, Z′ = randn(rng, N), randn(rng, N′), randn(rng, NZ), randn(rng, NZ′)

    # Construct projection with non-block matrices.
    f = GP(ConstantMean(3.0), EQ(), GPC())
    ϕ = lhsfinite(EQ(), Z)
    f_pr = project(ϕ, f(Z), sin)

    @test mean_vec(f_pr(X)) == pairwise(EQ(), X, Z) * mean_vec(f(Z)) + sin.(X)
    @test cov(f_pr(X)) ≈ pairwise(EQ(), X, Z) * cov(f(Z)) * pairwise(EQ(), X, Z)'
    @test maximum(abs.(xcov(f_pr(X), f(X′)) - pairwise(EQ(), X, Z) * xcov(f(Z), f(X′)))) < eps()
    @test maximum(abs.(xcov(f(X), f_pr(X′)) - xcov(f(X), f(Z)) * pairwise(EQ(), Z, X′))) < eps()

    # Construct projection with BlockGP as the thing over which we're projecting.
    Zb = BlockData([Z, Z′])
    ϕb = lhsfinite(BlockCrossKernel(reshape([EQ(), EQ()], :, 1)), Zb)
    f_pr_b = project(ϕb, f(Zb), cos)
 
    @test mean_vec(f_pr_b(X)) == pairwise(EQ(), Zb, X)' * mean_vec(f(Zb)) .+ cos.(X)
    ϕZX, ϕZX′, ϕZ = pairwise(EQ(), Zb, X), pairwise(EQ(), Zb, X′), cov(f(Zb))
    @test cov(f_pr_b(X)) == Xt_A_X(ϕZ, ϕZX)
    @test xcov(f_pr_b(X), f_pr_b(X′)) == Xt_A_Y(ϕZX, ϕZ, ϕZX′)
    @test xcov(f_pr_b(X), f(X′)) == ϕZX' * xcov(f(Zb), f(X′))
    @test xcov(f(X), f_pr_b(X′)) == xcov(f(X), f(Zb)) * ϕZX′

    # Check that Blocked version is project is consistent with dense.
    Zb2 = BlockData([Z[1:3], Z[4:5]])
    ϕb2 = lhsfinite(BlockCrossKernel(reshape([EQ(), EQ()], :, 1)), Zb2)
    f_pr_b2 = project(ϕb2, f(Zb2), sin)

    @test mean_vec(f_pr_b2(X)) ≈ mean_vec(f_pr(X))
    @test cov(f_pr_b2(X)) ≈ cov(f_pr(X))
    @test xcov(f_pr_b2(X), f_pr_b2(X′)) ≈ xcov(f_pr(X), f_pr(X′))
    @test xcov(f_pr_b2(X), f(X′)) ≈ xcov(f_pr(X), f(X′))
    @test xcov(f(X), f_pr_b2(X′)) ≈ xcov(f(X), f_pr(X′))
end

end
