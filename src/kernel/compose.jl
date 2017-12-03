import Base: show

"""
    Composite{V, O, T<:Tuple{Any, N} where N} <: Kernel{V}

A `Composite` kernel is generated through the application of `N`-ary operation `O` to a
collection of objects, one of which is assumed to be a kernel. The result is, of course, not
guaranteed to be a positive definite kernel. A `Composite` `Kernel` should therefore be
treated with a degree of care. It is the job of the caller to determine whether the
resulting pseudo-kernel is stationary or not.
"""
struct Composite{V, O<:Function, T<:Tuple{Any, N} where N} <: Kernel{V}
    args::T
end

struct LhsOp{O<:Function, Tf<:Function, Tk<:Kernel} <: Kernel{NonStationary}
    f::Tf
    k::Tk
end

struct RhsOp{O<:Function, Tk<:Kernel, Tf<:Function} <: Kernel{NonStationary}
    k::Tk
    f::Tf
end

for op in (:+, :*)
    T_op = typeof(eval(op))

    @eval begin

        # Composite defintions.
        $op(a::Real, b::Kernel) = $op(Constant(a), b)
        $op(a::Kernel, b::Real) = $op(a, Constant(b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel, Tb<:Kernel} =
            Composite{NonStationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel{Stationary}, Tb<:Kernel{Stationary}} =
            Composite{Stationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        (k::Composite{<:KernelType, $T_op})(x, y) = $op(k.args[1](x, y), k.args[2](x, y))
        show(io::IO, k::Composite{<:Any, $T_op}) =
            print(io, "Composite with operator $($op) and args $(k.args).")

        # LhsOp definitions.
        $op(f::Tf, k::Tk) where {Tf<:Function, Tk<:Kernel} = LhsOp{$T_op, Tf, Tk}(f, k)
        @inline (k::LhsOp{$T_op})(x, x′) = $op(k.f(x), k.k(x, x′))
        show(io::IO, k::LhsOp{$T_op}) = print(io, "LhsOp{$($T_op)}, f=$(k.f), k=$(k.k)")

        # RhsOp definitions.
        $op(k::Tk, f::Tf) where {Tk<:Kernel, Tf<:Function} = RhsOp{$T_op, Tk, Tf}(k, f)
        @inline (k::RhsOp{$T_op})(x, x′) = $op(k.k(x, x′), k.f(x′))
        show(io::IO, k::RhsOp{$T_op}) = print(op, "RhsOp{$($T_op)}, k=$(k.k), f=$(k.f)")
    end
end

==(a::T, b::T) where T<:Composite = a.args[1] == b.args[1] && a.args[2] == b.args[2]
==(a::T, b::T) where T<:Union{LhsOp, RhsOp} = a.k == b.k
