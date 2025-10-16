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
        val2 = prod((value(p.type, fullTensor[i, j], x[i]) for i in 1:nVariates))
        if !compeps(val1, val2, 1E-10)
            return false
        end
    end
    return true
end

function differentiatePolynomial(polType::PolynomialType, degree, nVariates, partial::Integer, x::AbstractVector{<:Real})
    @assert length(x) == nVariates "x must have size nVariates"
    p = Polynomial(degree, nVariates, polType)
    fullTensor = Array(p.tensor)
    for j in 1:size(p)
        val1 = derivative(p, x, j, partial)
        val2 = prod((i == partial ? derivative(p.type, fullTensor[i, j], x[i]) : value(p.type, fullTensor[i, j], x[i]) for i in 1:nVariates))
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
    @test evalPolynomial(Canonic, 4, nVariates, x)
    @test evalPolynomial(Hermite, 4, nVariates, x)
    @test evalPolynomial(Tchebychev, 4, nVariates, x)
end

@testset "Differentiate multivariate polynomials" begin
    degree = 4
    nVariates = 5
    partial = 3 # must be smaller than nVariates
    x = randn(nVariates)
    @test differentiatePolynomial(Canonic, 4, nVariates, partial, x)
    @test differentiatePolynomial(Hermite, 4, nVariates, partial, x)
    @test differentiatePolynomial(Tchebychev, 4, nVariates, partial, x)
end

@testset "Transformations" begin
    nSamples = 50000
    eps = 4. / sqrt(nSamples)
    x = [randn(5) for i in 1:nSamples]
    l = LinearTransformation(x)
    @test all([compeps(l.center[i], 0., eps) for i in 1:length(l.center)])
    @test all([compeps(l.scale[i], 1., sqrt(3) * eps) for i in 1:length(l.scale)])
end

@testset "Least squares" begin
    dim = 4
    deg = 3
    nSamples = 10000
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    data = [randn(dim) for i in 1:nSamples]
    y = f.(data)
    vslsq = VSLeastSquares{Float64}(Polynomial(deg, dim, Hermite), VoidTransformation())
    fit(vslsq, data, y)
    x = randn(dim)
    @test compeps(predict(vslsq, x), f(x), 1.E-10)
end