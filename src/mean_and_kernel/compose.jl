"""
    UnaryComposite <: Kernel

A composite kernel comprising a unary (scalar) operator `f` and one kernel `k`. A covariance
matrix is constructed by first constructing `cov(k, X, X′)`, then applying `f` element-wise
to this.
"""
struct UnaryComposite <: Kernel
    f::Any
    k::Kernel
end
isstationary(u::UnaryComposite) = isstationary(u.k)
(u::UnaryComposite)(x, x′) = u.f(u.k(x, x′))
cov(u::UnaryComposite, X::AM, X′::AM) = u.f.(cov(u.k, X, X′))

"""
    BinaryComposite <: Kernel

A composite kernel comprising a binary operator `f` and two other kernels `ka` and `kb`. A
covariance matrix is constructed by first constructing the covariance matrix for each kernel
`ka` and `kb`, and then combining them element-wise with f.
"""
struct BinaryComposite <: Kernel
    f::Any
    ka::Kernel
    kb::Kernel
end
isstationary(b::BinaryComposite) = isstationary(b.ka) && isstationary(b.kb)
(b::BinaryComposite)(x, x′) = b.f(b.ka(x, x′), b.kb(x, x′))
cov(b::BinaryComposite, X::AM, X′::AM) = b.f.(cov(b.ka, X, X′), cov(b.kb, X, X′))

for op in [:+, :*]
    @eval begin
        $op(k::Kernel, x::Real) = UnaryComposite(k->$op(k, x), k)
        $op(x::Real, k::Kernel) = UnaryComposite(k->$op(x, k), k)
        $op(k::Kernel, k′::Kernel) = BinaryComposite($op, k, k′)     
    end
end

# THIS MAYBE WORKS. IT'S NOT CURRENTLY TESTED AND IS THEREFORE DISABLED.
# """
#     MapReduce <: Kernel

# A composite kernel comprising a binary (scalar) reduction operator `f` and a vector of
# kernels `ks`. 
# """
# struct MapReduce <: Kernel
#     f::Any
#     ks::Vector{<:Kernel}
# end
# isfinite(k::MapReduce) = all(isfinite.(k.ks))
# isstationary(k::MapReduce) = all(isstationary.(k.ks))
# (k::MapReduce)(x, x′) = mapreduce(k->k(x, x′), k.f, f.ks)
# cov(k::MapReduce, X::AM, X′::AM) = mapreduce(k->cov(k, X, X′), (K, K′)->k.f.(K, K′), k.ks)

"""
    LhsOp <: CrossKernel

Return the binary function `g(x, x′) = op(f(x), k(x, x′))`.
"""
struct LhsOp <: CrossKernel
    op::Any
    f::Any
    k::CrossKernel
end
(k::LhsOp)(x, x′) = k.op(k.f(x), k.k(x, x′))
xcov(k::LhsOp, X::AM, X′::AM) = k.op.(k.f.(X), xcov(k.k, X, X′))

"""
    RhsOp <: CrossKernel

Defines a `CrossKernel` `g(x, x′) = op(k(x, x′), f(x′))`.
"""
struct RhsOp <: CrossKernel
    op::Any
    f::Any
    k::CrossKernel
end
(k::RhsOp)(x, x′) = k.op(k.k(x, x′), k.f(x′))
xcov(k::RhsOp, X::AM, X′::AM) = k.op.(xcov(k.k, X, X′), k.f.(X′))

# for op in (:+, :*)
#     T_op = typeof(eval(op))

#     @eval begin

#         # Composite defintions.
#         $op(a::Real, b::Kernel) = $op(Constant(a), b)
#         $op(a::Kernel, b::Real) = $op(a, Constant(b))
#         $op(a::Kernel, b::Kernel) = Composite{$T_op, Tuple{typeof(a), typeof(b)}}((a, b))
#         (k::Composite{$T_op})(x, y) = $op(k.args[1](x, y), k.args[2](x, y))
#         show(io::IO, k::Composite{$T_op}) = show(io, "$($op)($(k.args)...).")

#         # LhsOp definitions.
#         $op(f::Tf, k::Tk) where {Tf<:Function, Tk<:Kernel} = LhsOp{$T_op, Tf, Tk}(f, k)
#         @inline (k::LhsOp{$T_op})(x, x′) = $op(k.f(x), k.k(x, x′))
#         show(io::IO, k::LhsOp{$T_op}) = show(io, "LhsOp{$($T_op)}, f=$(k.f), k=$(k.k)")

#         # RhsOp definitions.
#         $op(k::Tk, f::Tf) where {Tk<:Kernel, Tf<:Function} = RhsOp{$T_op, Tk, Tf}(k, f)
#         @inline (k::RhsOp{$T_op})(x, x′) = $op(k.k(x, x′), k.f(x′))
#         show(io::IO, k::RhsOp{$T_op}) = show(op, "RhsOp{$($T_op)}, k=$(k.k), f=$(k.f)")
#     end
# end

# ==(a::T, b::T) where T<:Composite = a.args[1] == b.args[1] && a.args[2] == b.args[2]
# ==(a::T, b::T) where T<:Union{LhsOp, RhsOp} = a.k == b.k
