using Random, FillArrays
using Stheno: OneMean, EQ, PseudoPointsCov, TitsiasMean, TitsiasKernel

@testset "titsias" begin

    @testset "TitsiasKernel" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        z = collect(range(-3.0, 3.0; length=M))

        # THE EIGENVALUES ARE OFF. CHANGE THEM TO MAKE SENSE!
        A = randn(rng, M, N)
        S = PseudoPointsCov(cholesky(A * A' + I), cholesky(pw(EQ(), z)).U)
        k = TitsiasKernel(S, EQ(), EQ(), z)

        differentiable_kernel_tests(rng, k, x0, x1, x2)
    end




    # rng, N, N′, σ² = MersenneTwister(123456), 10, 13, 0.1
    # x, x′ = collect(range(-3.0, 3.0, length=N)), collect(range(-5.0, 5.0, length=10N))

    # x0 = collect(range(-3.0, stop=3.0, length=N))
    # x1 = collect(range(-3.5, stop=2.5, length=N))
    # x2 = collect(range(-2.5, stop=3.5, length=N′))

    # C = cholesky(pw(EQ(), x) + σ² * I)
    # k = EagerFinite(C)
    # @show k

    # @testset "ProjectedKernel" begin
    #     z = x
    #     k = ProjectedKernel(C, EQ(), z)
    #     differentiable_kernel_tests(rng, k, x0, x1, x2)
    # end

    # @testset "ProjectedCrossKernel" begin
    #     z = x
    #     k = ProjectedCrossKernel(C, EQ(), EQ(), z)
    #     differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
    # end
end
