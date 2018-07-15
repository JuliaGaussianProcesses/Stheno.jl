import Base: print, IO

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

function print(io::IO, x::Shunt)
    if x.n > 1
        print(io, x.n == 1 ? x.s : x.s^x.n)
    elseif x.n == 1
        print(io, x.s)
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

function print(io::IO, shunted::Shunted)
    print(io, shunted.shunt)
    print(io, shunted.x)
end

dummy_shunt(x) = Shunted(Shunt(0, ' '), x)

function print_shunted_list(io::IO, shunt::Shunt, X::AbstractVector)
    for (j, x) in enumerate(X)
        print(io, Shunted(shunt, x))
        j != length(X) && print('\n')
    end
end
@noinline print_shunted_list(io::IO, shunt::Shunt, x::Tuple) = print_shunted_list(io, shunt, [x...])
