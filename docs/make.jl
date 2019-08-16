using Documenter, Stheno

makedocs(
	modules = [Stheno],
    format = :html,
    sitename = "Stheno.jl",
    pages = [
        "Home" => "index.md",
        "Interfaces" => "mean_and_kernel_interfaces.md",
    ],
)
