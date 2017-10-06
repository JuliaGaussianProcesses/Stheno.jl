"""
    Composite{V, O, T<:Tuple{Kernel, N} where N} <: Kernel{V}

A `Composite` kernel is generated through the addition or multiplication of two existing
`Kernel` objects, or the addition or multiplication of an existing `Kernel` object and a
`Real`.
"""
struct Composite{V, O<:Function, T<:Tuple{Kernel, N} where N} <: Kernel{V}
    args::T
end

for op in (:+, :*)
    @eval begin
        $op(a::Real, b::Kernel) = $op(Constant(a), b)
        $op(a::Kernel, b::Real) = $op(a, Constant(b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel, Tb<:Kernel} =
            Composite{NonStationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel{Stationary}, Tb<:Kernel{Stationary}} =
            Composite{Stationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        (k::Composite{<:KernelType, typeof($op)})(x::T, y::T) where T =
            $op(k.args[1](x, y), k.args[2](x, y))
        function ==(
            a::Composite{<:KernelType, typeof($op), <:Tuple{Kernel, N} where N},
            b::Composite{<:KernelType, typeof($op), <:Tuple{Kernel, N} where N},
        )
            return (a.args[1] == b.args[1] && a.args[2] == b.args[2]) ||
                   (a.args[2] == b.args[1] && a.args[1] == b.args[2])
        end
    end
end
