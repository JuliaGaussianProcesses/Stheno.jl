@benchset "kernel" begin

    @benchset "ZeroKernel{Float64}()" begin
        create_benchmarks(ZeroKernel{Float64}(); grads=false)
    end
    @benchset "ConstantKernel(5.0)" begin
        create_benchmarks(ConstantKernel(5.0); grads=false)
    end
    @benchset "EQ" begin
        @benchset "EQ Real" create_benchmarks(EQ())

        for D in Ds()
            @benchset "EQ ColsAreObs (D=$D)" create_benchmarks(
                EQ();
                x=randn(D), x′=randn(D),
                x̄s=[ColsAreObs(randn(D, N)) for N in Ns()],
                x̄′s=[ColsAreObs(randn(D, N)) for N in Ns()],
            )
        end

        # See https://github.com/FluxML/Zygote.jl/issues/44
        # @benchset "EQ Almost-Toeplitz" create_benchmarks(
        #     EQ();
        #     x=randn(), x′=randn(),
        #     x̄s=[range(-randn(), step=randn(), length=N) for N in Ns()],
        #     x̄′s=[range(-randn(), step=randn(), length=N) for N in Ns()],
        # )

        # δ = randn()
        # @benchset "EQ Toeplitz" create_benchmarks(
        #     EQ();
        #     x=randn(), x′=randn(),
        #     x̄s=[range(-randn(), step=δ, length=N) for N in Ns()],
        #     x̄′s=[range(-randn(), step=δ, length=N) for N in Ns()],
        # )
    end
end
