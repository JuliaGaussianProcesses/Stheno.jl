@timedtestset "sparse gp" begin
    x = 0:0.1:10
    xu = 0:10
    σ = 1.0
    σu = 1e-3
    f = atomic(GP(Matern32Kernel()), GPC())
    covariance_error = "The covariance matrix of a sparse GP can often be dense and " *
        "can cause the computer to run out of memory. If you are sure you have enough " *
        "memory, you can use `cov(f.fobs)`."

    @timedtestset "SparseFiniteGP Constructors" begin
        f = atomic(GP(Matern32Kernel()), GPC())
        @test SparseFiniteGP(f(x), f(xu)) == SparseFiniteGP(f(x, 1e-18), f(xu, 1e-18))
    end

    @timedtestset "SparseFiniteGP methods" begin
        f = atomic(GP(Matern32Kernel()), GPC())
        fx = f(x)
        fxu = SparseFiniteGP(f(x), f(xu))
        @test mean(fxu) == mean(fx)
        @test marginals(fxu) == marginals(fx)
        @test rand(MersenneTwister(12345), fxu) == rand(MersenneTwister(12345), fx)
        @test_throws ErrorException(covariance_error) cov(fxu)
        @test cov(fxu.fobs) == cov(fx)
    end

    @timedtestset "SparseFiniteGP inference" begin
        f = atomic(GP(Matern32Kernel()), GPC())
        fx = f(x, σ)
        fxu = SparseFiniteGP(f(x, σ), f(xu, σu))
        y = rand(MersenneTwister(12345), fxu)

        fpost1 = posterior(VFE(fxu.finducing), fxu.fobs, y)
        fpost2 = posterior(fxu, y)

        @test marginals(fpost1(x)) == marginals(fpost2(x))
        @test elbo(fxu, y) == logpdf(fxu, y)
        @test logpdf(fxu, y) == elbo(VFE(fxu.finducing),fxu.fobs, y)
        yy = rand(fxu, 10)
        @test all(logpdf(fx, yy) .> logpdf(fxu, yy))
    end
end
