push!(LOAD_PATH,"../src/")

using Documenter, VectorSpaceLeastSquares, Changelog

# Generate a Documenter-friendly changelog from CHANGELOG.md
Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src", "release-notes.md");
    repo = "jlelong/VectorSpaceLeastSquares.jl",
)

makedocs(
    sitename="Vector Space Least Squares",
    modules = [VectorSpaceLeastSquares],
    doctest = false,
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Transformations" => "transformations.md",
            "Vector space basis" => "basis.md",
            "Least Squares problem" => "leastsquares.md",
        ],
        "Examples" => "examples.md",
        "Release notes" => "release-notes.md"
    ]
)
