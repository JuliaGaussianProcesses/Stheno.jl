using Stheno: UnaryMean, BinaryMean, BinaryKernel, BinaryCrossKernel, LhsCross, RhsCross,
    OuterCross, OuterKernel

@benchset "compose" begin
    let
        m1, m2 = CustomMean(sin), CustomMean(cos)
        @benchset "UnaryMean CPU" create_benchmarks(UnaryMean(exp, m1))
        @benchset "BinaryMean CPU" create_benchmarks(BinaryMean(+, m1, m2))
    end
    let
        k1, k2 = SEKernel(), OneKernel()
        @benchset "BinaryKernel SEKernel CPU" create_benchmarks(BinaryKernel(*, k1, k1))
        @benchset "BinaryKernel Const. CPU" create_benchmarks(BinaryKernel(*, k2, k2))
    end
    let
        k, f = SEKernel(), abs2
        @benchset "LhsCross abs2 * SEKernel CPU" create_benchmarks(LhsCross(f, k))
        @benchset "RhsCross SEKernel * abs2 CPU" create_benchmarks(RhsCross(k, f))
        @benchset "OuterCross f * SEKernel * f CPU" create_benchmarks(OuterCross(f, k))
        @benchset "OuterKernel f * SEKernel * f CPU" create_benchmarks(OuterKernel(f, k))
    end
end
