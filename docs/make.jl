using Documenter, Stheno

makedocs(
	modules = [Stheno],
    format = Documenter.HTML(),
    sitename = "Stheno.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Internals" => "internals.md",
        "Input Types" => "input_types.md",
    ],
)

deploydocs(repo="github.com/willtebbutt/Stheno.jl.git")
