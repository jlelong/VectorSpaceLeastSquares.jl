using VectorSpaceLeastSquares
using Test
using SparseArrays

@testset "VectorSpaceLeastSquares.jl" begin
    # Write your tests here.
    @test VectorSpaceLeastSquares.Polynomial(2, 3, VectorSpaceLeastSquares.Canonic).tensor == sparse([3, 3, 2, 2, 3, 2, 1, 1, 3, 1, 2, 1], [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9], [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2], 3, 10)
end
