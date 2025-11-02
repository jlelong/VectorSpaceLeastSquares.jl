if false; include("../src/VectorSpaceLeastSquares.jl"); end
using VectorSpaceLeastSquares
using Test
using SparseArrays

function compeps(a::Real, b::Real, eps::Real)
    return abs(b - a) < eps
end

function compeps(a::Vector{<:Real}, b::Vector{<:Real}, eps::Real)
    @assert length(a) == length(b) "a and b must have the same length"
    return all((compeps(ai, bi, eps) for (ai, bi) in zip(a, b)))
end

function createPolynomial()
    return getTensor(PolynomialBasis(Canonic, 3, 2)) == sparse([3, 3, 2, 2, 3, 2, 1, 1, 3, 1, 2, 1], [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9], [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2], 3, 10)
end

function testPol1d(degree::Integer, x::Real, func1d::Function)
    @assert degree >= 2 "degree must be >= 2"
    f1 = func1d(x, 1)
    f2 = func1d(x, 2)
    compeps(func1d(x, degree), func1d(x, degree, 2, f2, f1), 1.E-10)
end

function evalPolynomial(polType::PolynomialType, degree, nVariates, x::AbstractVector{<:Real})
    @assert length(x) == nVariates "x must have size nVariates"
    p = PolynomialBasis(polType, nVariates,degree)
    fullTensor = Array(getTensor(p))
    for j in 1:length(p)
        val1 = value(p, x, j)
        val2 = prod((value(getType(p), fullTensor[i, j], x[i]) for i in 1:nVariates))
        if !compeps(val1, val2, 1E-10)
            return false
        end
    end
    return true
end

function differentiatePolynomial(polType::PolynomialType, degree, nVariates, partial::Integer, x::AbstractVector{<:Real})
    @assert length(x) == nVariates "x must have size nVariates"
    p = PolynomialBasis(polType, nVariates,degree)
    fullTensor = Array(getTensor(p))
    for j in 1:length(p)
        val1 = derivative(p, x, j, partial)
        val2 = prod((i == partial ? derivative(getType(p), fullTensor[i, j], x[i]) : value(getType(p), fullTensor[i, j], x[i]) for i in 1:nVariates))
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

function testLinearTransformation()
    dim = 3
    nSamples = 50000
    eps = 4. / sqrt(nSamples)
    x = [randn(dim) for i in 1:nSamples]
    t = LinearTransformation(x)
    @test all([compeps(t.center[i], 0., eps) for i in 1:length(t.center)])
    @test all([compeps(t.scale[i], 1., sqrt(3) * eps) for i in 1:length(t.scale)])
end

function testGaussianTransformation()
    nSamples = 50000
    dim = 3
    mean = 1
    sigma = 2
    eps = 4. / sqrt(nSamples)
    x = [mean .+ sigma .* randn(dim) for i in 1:nSamples]
    t = GaussianTransformation(x)
    @test all([compeps(t.mean[i], mean, sigma * eps) for i in 1:length(t.mean)])
    @test all([compeps(t.sigma[i], sigma, sqrt(3) * sigma^2 * eps) for i in 1:length(t.sigma)])
end

function testLogNormalTransformation()
    dim = 3
    nSamples = 50000
    mean = 1
    sigma = 2
    eps = 4. / sqrt(nSamples)
    x = [exp.(mean .+ sigma .* randn(dim)) for i in 1:nSamples]
    t = LogNormalTransformation(x)
    @test all([compeps(t.mean[i], mean, sigma * eps) for i in 1:length(t.mean)])
    @test all([compeps(t.sigma[i], sigma, sqrt(3) * sigma^2 * eps) for i in 1:length(t.sigma)])
end

@testset "Transformations" begin
    testLinearTransformation()
    testGaussianTransformation()
    testLogNormalTransformation()
end

function testFitVoidTransformationPolynomialBasis(T::Type, eps)
    dim = 4
    deg = 3
    nSamples = 10000
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    data = [randn(T, dim) for i in 1:nSamples]
    y = f.(data)
    vslsq = VSLeastSquares(PolynomialBasis(Hermite, dim, deg), VoidTransformation(), T)
    fit(vslsq, data, y)
    x = randn(T, dim)
    return compeps(predict(vslsq, x), f(x), T(eps))
end

function testFitLinearTransformationPolynomialBasis(T::Type, eps)
    dim = 4
    deg = 3
    nSamples = 10000
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    df(x) = [
        - x[4] + 7 * x[3]^2,
        6 * x[2]^2,
        14 * x[3] * x[1],
        - x[1]
    ]
    data = [1.0 .+ 2.0 .* randn(T, dim) for i in 1:nSamples]
    y = f.(data)
    transformation = LinearTransformation(data)
    vslsq = VSLeastSquares(PolynomialBasis(Hermite, dim, deg), transformation, T)
    fit(vslsq, data, y)
    x = randn(T, dim)
    @test compeps(predict(vslsq, x), f(x), T(eps))
    @test compeps(gradient(vslsq, x), df(x), T(eps))
end

function testVoidTransformationPiecewiseConstantBasis(T::Type, dim, eps)
    nIntervals = 50
    nSamples = 100000
    data = [rand(T, dim) for i in 1:nSamples]
    f(x) = log(1. + sum(x.^2))
    y = f.(data)
    vslsq = VSLeastSquares(PiecewiseConstantBasis(dim, nIntervals), VoidTransformation(), T)
    fit(vslsq, data, y)
    x = rand(T, dim)
    @test compeps(predict(vslsq, x), f(x), T(eps))
end

@testset "Least squares" begin
    @test testFitVoidTransformationPolynomialBasis(Float32, 1.E-3)
    @test testFitVoidTransformationPolynomialBasis(Float64, 1.E-10)
    testFitLinearTransformationPolynomialBasis(Float64, 1.E-3)
    testVoidTransformationPiecewiseConstantBasis(Float64, 1, 1.E-2)
    testVoidTransformationPiecewiseConstantBasis(Float64, 2, 1.E-2)
end