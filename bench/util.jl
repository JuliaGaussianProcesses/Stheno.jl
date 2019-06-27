Ns() = [1_000,]
# Ns() = [10,]
Ds() = [1,]

generate_x̄s(x, x̄s::Nothing) = [fill(x, N) for N in Ns()]
generate_x̄s(x, x̄s) = x̄s

function create_benchmarks(name, grads::Bool, f, x...)
    if grads
        @benchset name begin
            @bench "eval" $f($x...)
            __forward() = Zygote.forward(f, x...)
            out, back = Zygote.forward(f, x...)
            @bench "forward" $__forward()
            @bench "back" $back($out)
            # @bench "gradient" $__gradient()
        end
    else
        @bench name $f($x...)
    end
    return nothing
end

# Common benchmarks to run for mean functions.
function create_benchmarks(μ::MeanFunction; grads=true, x=5.0, x̄s=nothing)
    create_benchmarks("μ(x)", grads, μ, x)

    for x̄ in generate_x̄s(x, x̄s)
        N = length(x̄)
        create_benchmarks("map ($(length(x̄)))", grads, x->map(μ, x), x̄)
    end
end

# Common benchmarks for CrossKernel.
function create_benchmarks(k::CrossKernel; x=5.0, x′=4.0, x̄s=nothing, x̄′s=nothing, grads=true)
    create_benchmarks("k(x, x′)", grads, k, x, x′)

    for (x̄, x̄′) in zip(generate_x̄s(x, x̄s), generate_x̄s(x′, x̄′s))
        N = length(x̄)
        @benchset "$N" begin
            create_benchmarks("map k(x̄, x̄′) $N", grads, (x̄, x̄′)->map(k, x̄, x̄′), x̄, x̄′)
            create_benchmarks("pw k(x̄, x̄′) $N", grads, (x̄, x̄′)->pairwise(k, x̄, x̄′), x̄, x̄′)
        end
    end
end

# Common benchmarks to run for Kernels.
function create_benchmarks(k::Kernel; x=5.0, x′=4.0, x̄s=nothing, x̄′s=nothing, grads=true)
    create_benchmarks("k(x)", grads, k, x)
    create_benchmarks("k(x, x′)", grads, k, x, x′)

    for (x̄, x̄′) in zip(generate_x̄s(x, x̄s), generate_x̄s(x′, x̄′s))
        N = length(x̄)
        @benchset "$N" begin
            create_benchmarks("map k(x̄) $N", grads, x̄->map(k, x̄), x̄)
            create_benchmarks("map k(x̄, x̄′) $N", grads, (x̄, x̄′)->map(k, x̄, x̄′), x̄, x̄′)
            create_benchmarks("pw k(x̄) $N", grads, x̄->pairwise(k, x̄), x̄)
            create_benchmarks("pw k(x̄, x̄′) $N", grads, (x̄, x̄′)->pairwise(k, x̄, x̄′), x̄, x̄′)
        end
    end
end

function pretty_print(d::BenchmarkGroup, pre=1)
    pretty_print(d.data, pre)
end

function pretty_print(d::Dict, pre=1)
    for (k, v) in d
        if v isa BenchmarkGroup
            s = "$(repr(k))"
            println(join(fill(" ", pre)) * s)
            pretty_print(v, pre+1+4)
        else
            println(join(fill(" ", pre)) * "$(repr(k)) => $(repr(v))")
        end
    end
    nothing
end
