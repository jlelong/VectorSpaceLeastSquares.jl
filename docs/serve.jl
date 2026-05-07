push!(LOAD_PATH,"src/", "../src/")
using LiveServer
using VectorSpaceLeastSquares
servedocs(verbose=true, skip_files=[joinpath("docs", "src", "release-notes.md")])
