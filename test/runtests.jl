if false; include("../src/VectorSpaceLeastSquares.jl"); end
using VectorSpaceLeastSquares
using Test
using SparseArrays

function compeps(a::Real, b::Real, eps::Real)
    return abs(b - a) < eps
end

function createPolynomial()
    return VectorSpaceLeastSquares.Polynomial(2, 3, VectorSpaceLeastSquares.Canonic).tensor == sparse([3, 3, 2, 2, 3, 2, 1, 1, 3, 1, 2, 1], [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9], [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2], 3, 10)
end

function testPol1d(degree::Integer, x::Real, func1d::Function)
    @assert degree >= 2 "degree must be >= 2"
    f0 = func1d(x, 0)
    f1 = func1d(x, 1)
    compeps(func1d(x, degree), func1d(x, degree, 1, f1, f0), 1.E-10)
end

@testset "Evaluate 1d polynomials" begin
    @test testPol1d(4, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(2, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(6, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(4, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testPol1d(2, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testPol1d(6, 3., VectorSpaceLeastSquares.tchebychev1d)
end


@testset "Create multivariate polynomials" begin
    # Write your tests here.
    @test createPolynomial()
end
