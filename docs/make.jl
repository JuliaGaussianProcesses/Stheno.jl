using Documenter, Stheno

makedocs(
	modules = [Stheno],
    format = :html,
    sitename = "Stheno.jl",
    pages = [
        "Home" => "index.md",
        "BlockArrays extensions" => "block_arrays_ext.md",
        "Interfaces" => "mean_and_kernel_interfaces.md",
    ],
)
