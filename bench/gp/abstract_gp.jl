@benchset "abstract_gp" begin
    for N in [10,]
        f = GP(CustomMean(sin), SEKernel(), GPC())
        x = randn(N)
        fx = f(x)
        y = rand(fx)
        @benchset "N=$N" begin
            create_benchmarks("logpdf SEKernel $N (x, y)", true, (x, y)->logpdf(f(x), y), x, y)
            create_benchmarks("rand SEKernel $N", true, x->rand(f(x)), x)
            create_benchmarks("cov(f(x)) $N", true, x->cov(f(x)), x)
        end
    end
end
