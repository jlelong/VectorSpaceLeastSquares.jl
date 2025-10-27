push!(LOAD_PATH,"../src/")

using Documenter, VectorSpaceLeastSquares

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
        "Examples" => "examples.md"
    ]
)
