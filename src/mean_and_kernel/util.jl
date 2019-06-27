import Base: exp
export eq, linear, exp, noise

for (k_name, k_type) in [(:eq, :EQ), (:linear, :Linear), (:exp, :Exp)]
    @eval function $k_name(;A=nothing, l=nothing, α=nothing, β=nothing)
        k = $k_type()
        k = isnothing(A) ? k : transform(k, LinearTransform(A))
        k = isnothing(l) ? k : transform(k, Stretch(l))
        k = isnothing(α) ? k : OuterKernel(ConstMean(α), k)
        k = isnothing(β) ? k : k + ConstKernel(β)
        return k
    end 
end

function noise(;α=nothing, β=nothing)
    k = Noise()
    k = isnothing(α) ? k : OuterKernel(ConstMean(α), k)
    k = isnothing(β) ? k : k + ConstKernel(β)
    return k
end
