    # # Test logpdf + elbo do something vaguely sensible + that elbo converges to logpdf.
    # let
    #     rng, N, N′, D, gpc = MersenneTwister(123456), 25, 10, 2, GPC()
    #     X, X′ = rand(rng, D, N), rand(rng, D, N′)
    #     f = GP(ConstantMean(1.0), EQ(), gpc) + GP(ZeroMean{Float64}(), Noise(1e-1), gpc)
    #     y, y′ = rand(rng, f, X), rand(rng, f, X′)

    #     logpdf_y = logpdf([f], [X], BlockVector([y]))
    #     logpdf_yy′ = logpdf([f, f], [X, X′], BlockVector([y, y′]))

    #     @test logpdf_y isa Real
    #     @test logpdf(f, X, y) == logpdf_y
    #     @test logpdf_yy′ isa Real

    #     elbo_y = elbo([f], [X], BlockVector([y]), [f], [X], 2e-5)
    #     elbo_yy′ = elbo([f, f], [X, X′], BlockVector([y, y′]), [f, f], [X, X′], 2e-5)

    #     @test elbo_y < logpdf_y
    #     @test elbo_yy′ < logpdf_yy′
    # end
