@benchset "finite" begin
    @benchset "FiniteMean" begin
        for N in Ns()
            μ = FiniteMean(CustomMean(sin), randn(N))
            @benchset(
                "CustomMean(sin) $N",
                create_benchmarks(μ; grads=false, x=1, x̄s=[1:N-1, 1:N,]),
            )
        end
    end
    @benchset "FiniteKernel" begin
        for N in Ns()
            k = FiniteKernel(SEKernel(), randn(N))
            @benchset(
                "SEKernel() $N",
                create_benchmarks(k; grads=false, x=1, x′=2, x̄s=[1:N,], x̄′s=[1:N,]),
            )
        end
    end
    @benchset "LhsFiniteCrossKernel" begin
        for N in Ns()
            k = LhsFiniteCrossKernel(SEKernel(), randn(N))
            @benchset(
                "SEKernel() $N",
                create_benchmarks(k; grads=false, x=1, x′=2.0, x̄s=[1:N,], x̄′s=[randn(N),]),
            )
        end
    end
    @benchset "RhsFiniteCrossKernel" begin
        for N in Ns()
            k = RhsFiniteCrossKernel(SEKernel(), randn(N))
            @benchset(
                "SEKernel() $N",
                create_benchmarks(k; grads=false, x=1.0, x′=2, x̄s=[randn(N),], x̄′s=[1:N,]),
            )
        end
    end
    @benchset "FiniteCrossKernel" begin
        for N in Ns()
            k = FiniteCrossKernel(SEKernel(), randn(N), randn(N))
            @benchset(
                "SEKernel() $N",
                create_benchmarks(k; grads=false, x=1, x′=2, x̄s=[1:N,], x̄′s=[1:N,]),
            )
        end
    end
end
