using Stheno: UnaryMean, BinaryMean, BinaryKernel, BinaryCrossKernel, LhsCross, RhsCross,
    OuterCross, OuterKernel

@benchset "compose" begin
    let
        m1, m2 = CustomMean(sin), CustomMean(cos)
        @benchset "UnaryMean CPU" create_benchmarks(UnaryMean(exp, m1))
        @benchset "BinaryMean CPU" create_benchmarks(BinaryMean(+, m1, m2))
    end
    let
        k1, k2 = EQ(), OneKernel()
        @benchset "BinaryKernel EQ CPU" create_benchmarks(BinaryKernel(*, k1, k1))
        @benchset "BinaryKernel Const. CPU" create_benchmarks(BinaryKernel(*, k2, k2))
    end
    let
        k, f = EQ(), abs2
        @benchset "LhsCross abs2 * EQ CPU" create_benchmarks(LhsCross(f, k))
        @benchset "RhsCross EQ * abs2 CPU" create_benchmarks(RhsCross(k, f))
        @benchset "OuterCross f * EQ * f CPU" create_benchmarks(OuterCross(f, k))
        @benchset "OuterKernel f * EQ * f CPU" create_benchmarks(OuterKernel(f, k))
    end
end
