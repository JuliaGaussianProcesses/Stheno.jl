using Pkg
Pkg.develop(path=joinpath(@__DIR__, ".."))

using Documenter
using Stheno

### Process examples

# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")

# State explicitly which examples are to be included, rather than including everything by
# default. This can make the docs development cycle faster -- it means that if you're not
# working on the examples then you can just make `examples` an empty vector after building
# all of the examples in a first run of the docs.
# Similarly, when developing only a subset of the examples, only a subset need to be
# re-built every time. To achieve this, comment out those examples that you don't wish to
# repeatedly re-build. Alternatively, if you're just working on a single example, either
# work on the source directly, or work with the literate.jl file to avoid needing to start
# a fresh Julia session each time you want to run the example.
examples = [
    "getting_started",
    "custom_affine_transformations",
    "process_decomposition",
    "sensor_fusion",
    "time_varying_blr",
    "gppp_and_pseudo_points",
    "differentiation",
    "extended_mauna_loa",
    "quadrature-convolution",
]

example_locations = map(example -> joinpath(@__DIR__, "..", "examples", example), examples)

if ispath(EXAMPLES_OUT)
    map(examples) do example
        path = joinpath(EXAMPLES_OUT, example)
        println(path)
        isdir(path) && rm(path; recursive=true)
    end
else
    mkpath(EXAMPLES_OUT)
end

dev_command = "Pkg.develop(PackageSpec(; path=relpath(\"$(pkgdir(Stheno))\", pwd())));"

let script = "using Pkg; Pkg.activate(ARGS[1]); $dev_command Pkg.instantiate()"
    for example in example_locations
        if !success(`$(Base.julia_cmd()) -e $script $example`)
            error(
                "project environment of example ",
                basename(example),
                " could not be instantiated",
            )
        end
    end
end

# Run examples asynchronously
literate_path = joinpath(@__DIR__, "literate.jl")
processes = map(example_locations) do example
    return run(
        pipeline(
            `$(Base.julia_cmd()) $literate_path $(basename(example)) $EXAMPLES_OUT`;
            stdin=devnull,
            stdout=devnull,
            stderr=stderr,
        );
        wait=true,
    )::Base.Process
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")



### Build documentation
DocMeta.setdocmeta!(
    Stheno,
    :DocTestSetup,
    :(using Stheno.AbstractGPs, Stheno, Random, LinearAlgebra);
    recursive=true,
)

makedocs(
    modules=[Stheno],
    format=Documenter.HTML(),
    sitename="Stheno.jl",
    pages=[
        "Home" => "index.md",
        "Getting Started" => joinpath("examples", "getting_started.md"),
        "Input Types" => "input_types.md",
        "Kernel Design" => "kernel_design.md",
        "Internals" => "internals.md",
        "Custom Affine Transformations" =>
            joinpath("examples", "custom_affine_transformations.md"),
        "API" => "api.md",
        "Examples" => [
            "examples_note.md",
            joinpath("examples", "process_decomposition.md"),
            joinpath("examples", "sensor_fusion.md"),
            joinpath("examples", "time_varying_blr.md"),
            joinpath("examples", "gppp_and_pseudo_points.md"),
            joinpath("examples", "extended_mauna_loa.md"),
        ],
        "Experimental Examples" => [
            "experimental_examples_notes.md",
            joinpath("examples", "differentiation.md"),
            joinpath("examples", "quadrature-convolution.md"),
        ],
    ],
    doctestfilters=[
        r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
        r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
        r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
    ],
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/Stheno.jl.git", push_preview=true)
