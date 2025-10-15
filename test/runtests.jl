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
    f1 = func1d(x, 1)
    f2 = func1d(x, 2)
    compeps(func1d(x, degree), func1d(x, degree, 2, f2, f1), 1.E-10)
end

function evalPolynomial(polType::PolynomialType, degree, nVariates, x::AbstractVector{<:Real})
    @assert length(x) == nVariates "x must have size nVariates"
    p = Polynomial(degree, nVariates, polType)
    fullTensor = Array(p.tensor)
    for j in 1:size(p)
        val1 = value(p, x, j)
        val2 = prod([value(p.type, fullTensor[i, j], x[i]) for i in 1:nVariates])
        if !compeps(val1, val2, 1E-10)
            return false
        end
    end
    return true
end

@testset "Evaluate 1d polynomials" begin
    @test testPol1d(4, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(2, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(6, 3., VectorSpaceLeastSquares.hermite1d)
    @test testPol1d(4, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testPol1d(2, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testPol1d(6, 3., VectorSpaceLeastSquares.tchebychev1d)
end

@testset "Differentiate 1d polynomials" begin
    @test testPol1d(4, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testPol1d(2, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testPol1d(6, 3., VectorSpaceLeastSquares.dtchebychev1d)
end


@testset "Create multivariate polynomials" begin
    # Write your tests here.
    @test createPolynomial()
end

@testset "Evaluate multi-variate polynomials" begin
    degree = 4
    nVariates = 5
    x = randn(nVariates)
    evalPolynomial(Canonic, 4, nVariates, x)
    evalPolynomial(Hermite, 4, nVariates, x)
    evalPolynomial(Tchebychev, 4, nVariates, x)
end

@testset "Differentiate multivariate polynomials" begin
    
end