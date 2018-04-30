# @testset "integrate" begin

#     # Test basic properties of the prior.
#     let f = GP(ZeroMean(), EQ(), GPC())
#         v = ∫(StandardNormal(), f)
#         @test mean(v)(1) == 0.0
#         @test mean_vector(v) == [0.0]
#         @test kernel(v)(1, 1) == 1 / sqrt(3)
#         @test kernel(v, f)(1, 5.0) == kernel(f, v)(5.0, 1)
#         @test kernel(v, f)(1, 0.0) == 1 / sqrt(2)
#     end

#     # Test properties of the posterior when the integral is observed.
#     let f = GP(ZeroMean(), EQ(), GPC())
#         v = ∫(StandardNormal(), f)
#         v′, f′ = (v | (v←[1])), (f | (v←[1]))
#         x = collect(range(-10.0, stop=10.0, length=100))
#         g(x) = exp(-0.5 * x^2) / sqrt(2π)
#         @test abs(quadgk(x->mean(f′)(x) * g(x), -10.0, 10.0)[1] - 1.0) < 1e-6
#     end

#     # Test properties of the posterior when the function is observed.
#     let rng = MersenneTwister(123456), f = GP(ZeroMean(), EQ(), GPC())
#         v = ∫(StandardNormal(), f)
#         x = collect(range(-10.0, stop=10.0, length=100))
#         fs = sample(rng, f(x))
#         v′, μf′ = mean(v | (f(x)←fs))(1), mean(f | (f(x)←fs))
#         g(x) = exp(-0.5 * x^2) / sqrt(2π)
#         @test abs(quadgk(x->μf′(x) * g(x), -10.0, 10.0)[1] - v′) < 1e-6
#     end
# end
