using Stheno: DerivativeMean, DerivativeKernel, DerivativeLhsCross, DerivativeRhsCross,
    DerivativeCross
using Stheno: EQ, Exp, Linear, Noise, PerEQ

@testset "derivative" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)

        mean_function_tests(DerivativeMean(CustomMean(sin)), x0)
        kernel_tests(DerivativeKernel(EQ()), x0, x1, x2)
        cross_kernel_tests(DerivativeLhsCross(EQ()), x0, x1, x2)
        cross_kernel_tests(DerivativeRhsCross(EQ()), x0, x1, x2)
        cross_kernel_tests(DerivativeCross(EQ()), x0, x1, x2)
    end
end
