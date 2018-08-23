import Base: show, IO

"""
    Shunt

Print `s` `n` times before printing `x`.
"""
struct Shunt
    n::Int
    s::Char
    function Shunt(n::Int, s::Char)
        @assert n >= 0
        return new(n, s)
    end
end

function show(io::IO, x::Shunt)
    if x.n > 1
        show(io, x.n == 1 ? x.s : x.s^x.n)
    elseif x.n == 1
        show(io, x.s)
    end
end

extend(shunt::Shunt, n::Int=4) = Shunt(shunt.n + n, shunt.s)

"""
    Shunted{Tx}

Apply `shunt` before printing `x`.
"""
struct Shunted{Tx}
    shunt::Shunt
    x::Tx
end

function show(io::IO, shunted::Shunted)
    show(io, shunted.shunt)
    show(io, shunted.x)
end

dummy_shunt(x) = Shunted(Shunt(0, ' '), x)

function print_shunted_list(io::IO, shunt::Shunt, X::AbstractVector)
    for (j, x) in enumerate(X)
        show(io, Shunted(shunt, x))
        j != length(X) && show('\n')
    end
end
@noinline function print_shunted_list(io::IO, shunt::Shunt, x::Tuple)
    return print_shunted_list(io, shunt, [x...])
end
