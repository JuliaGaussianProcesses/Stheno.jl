"""
    additive_gp(fs)

Produces the GP given by
```julia
sum(fs[1](x[1]) + fs[2](x[2]) + ... + fs[D](x[D]))
```
Requires that `length(fs)` is the same as the dimension of the inputs to be used.
"""
additive_gp(fs) = additive_gp(fs, 1:length(fs))

"""
    additive_gp(fs, indices)

`fs` should be a collection of GPs, and `indices` a collection of collections of integer
indices.
For example, `indices` might be something like `[1:2, 3, 4:6]`, in which case `fs` would
need to comprise exactly three elements. In general, this functions requires that
`length(fs) == length(indices)`.

Produces the GP given by
```julia
sum(fs[1](x[indices[1]]) + fs[2](x[indices[2]]) + ... + fs[D](x[indices[D]]))
```
"""
function additive_gp(fs, indices)
    fs_projected = map((f, idx) -> f ∘ Select(idx), fs, indices)
    return sum(fs_projected)
end
