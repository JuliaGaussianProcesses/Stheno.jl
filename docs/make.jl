using Documenter, Stheno

DocMeta.setdocmeta!(
    Stheno,
    :DocTestSetup,
    :(using AbstractGPs, Stheno, Random, LinearAlgebra);
    recursive=true,
)

makedocs(
	modules = [Stheno],
    format = Documenter.HTML(),
    sitename = "Stheno.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "CompositeGP API" => "composite_gp_api.md",
        "Input Types" => "input_types.md",
        "Kernel Design" => "kernel_design.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(repo="github.com/willtebbutt/Stheno.jl.git")
