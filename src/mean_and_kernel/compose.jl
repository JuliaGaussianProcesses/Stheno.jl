import Base: show

"""
    Composite{O, T<:Tuple{Kernel, N} where N} <: Kernel

A `Composite` kernel is generated through the application of `N`-ary operation `O` to a
collection of objects, at least one of which is assumed to be a kernel.
"""
struct Composite{O<:Function, T<:Tuple{Vararg{Kernel}}} <: Kernel
    args::T
end
for foo in [:isfinite, :isstationary]
    @eval function $foo(::Type{<:Composite{<:Function, T}}) where T<:Tuple{Vararg{Kernel}}
        for n in 1:length(T.parameters)
            !$foo(T.parameters[n]) && return false
        end
        return true
    end
end

"""
    LhsOp{O<:Function, Tf<:Function, Tk<:Kernel}

Return the binary function `g(x, x′) = O(f(x), k(x, x′))`.
"""
struct LhsOp{O<:Function, Tf<:Function, Tk<:Kernel} <: Kernel
    f::Tf
    k::Tk
end

"""
    RhsOp{O<:Function, Tk<:Kernel, Tf<:Function}

Return the binary fucntion `g(x, x′) = O(k(x, x′), f(x′))`.
"""
struct RhsOp{O<:Function, Tk<:Kernel, Tf<:Function} <: Kernel
    k::Tk
    f::Tf
end

for op in (:+, :*)
    T_op = typeof(eval(op))

    @eval begin

        # Composite defintions.
        $op(a::Real, b::Kernel) = $op(Constant(a), b)
        $op(a::Kernel, b::Real) = $op(a, Constant(b))
        $op(a::Kernel, b::Kernel) = Composite{$T_op, Tuple{typeof(a), typeof(b)}}((a, b))
        (k::Composite{$T_op})(x, y) = $op(k.args[1](x, y), k.args[2](x, y))
        show(io::IO, k::Composite{$T_op}) = show(io, "$($op)($(k.args)...).")

        # LhsOp definitions.
        $op(f::Tf, k::Tk) where {Tf<:Function, Tk<:Kernel} = LhsOp{$T_op, Tf, Tk}(f, k)
        @inline (k::LhsOp{$T_op})(x, x′) = $op(k.f(x), k.k(x, x′))
        show(io::IO, k::LhsOp{$T_op}) = show(io, "LhsOp{$($T_op)}, f=$(k.f), k=$(k.k)")

        # RhsOp definitions.
        $op(k::Tk, f::Tf) where {Tk<:Kernel, Tf<:Function} = RhsOp{$T_op, Tk, Tf}(k, f)
        @inline (k::RhsOp{$T_op})(x, x′) = $op(k.k(x, x′), k.f(x′))
        show(io::IO, k::RhsOp{$T_op}) = show(op, "RhsOp{$($T_op)}, k=$(k.k), f=$(k.f)")
    end
end

==(a::T, b::T) where T<:Composite = a.args[1] == b.args[1] && a.args[2] == b.args[2]
==(a::T, b::T) where T<:Union{LhsOp, RhsOp} = a.k == b.k
