using Stheno: UnaryMean, BinaryMean, BinaryKernel, BinaryCrossKernel

@benchset "compose" begin
    # let
    #     m1, m2 = CustomMean(sin), CustomMean(cos)
    #     @benchset "UnaryMean CPU" create_benchmarks(UnaryMean(exp, m1))
    #     @benchset "BinaryMean CPU" create_benchmarks(BinaryMean(+, m1, m2))
    # end
    let
        k1, k2 = EQ(), ZeroKernel{Float64}()
        @benchset "BinaryKernel EQ CPU" create_benchmarks(BinaryKernel(*, k1, k1); grads=false)
        @benchset "BinaryKernel 0 CPU" create_benchmarks(BinaryKernel(*, k2, k2); grads=false)
    end
end
