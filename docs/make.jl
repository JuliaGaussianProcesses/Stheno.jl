using Documenter, Stheno

makedocs(
	modules = [Stheno],
    format = :html,
    sitename = "Stheno.jl",
    pages = [
        "Home" => "index.md",
        "Internals" => "internals.md",
    ],
)
