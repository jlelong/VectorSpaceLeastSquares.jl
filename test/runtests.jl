if false; include("../src/VectorSpaceLeastSquares.jl"); end
using VectorSpaceLeastSquares
using Test
using SparseArrays
using Distributions: cdf, pdf, Normal


function compeps(actual::Real, expected::Real, eps::Real)
    t = abs(actual - expected) < eps
    if !t
        println("expected: $(expected)")
        println("actual: $(actual)")
    end
    return t
end

function compeps(actual::AbstractVector{<:Real}, expected::AbstractVector{<:Real}, eps::Real)
    @assert length(expected) == length(actual) "both vectors must have the same length"
    t = all((compeps(ai, bi, eps) for (ai, bi) in zip(expected, actual)))
    if !t
        println("expected: $(expected)")
        println("actual: $(actual)")
    end
    return t
end

function compeps(actual::AbstractMatrix{<:Real}, expected::AbstractMatrix{<:Real}, eps::Real)
    @assert size(expected) == size(actual) "both matrices must have the same size"
    t = all((compeps(ai, bi, eps) for (ai, bi) in zip(expected, actual)))
    if !t
        println("expected: $(expected)")
        println("actual: $(actual)")
    end
    return t
end


function createPolynomial()
    return getTensor(PolynomialBasis(Canonic, 3, 2)) == sparse([3, 3, 2, 2, 3, 2, 1, 1, 3, 1, 2, 1], [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9], [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2], 3, 10)
end

function testRecursiveDerivative(degree::Integer, x::Real, func1d::Function)
    @assert degree >= 2 "degree must be >= 2"
    f1 = func1d(x, 1)
    f2 = func1d(x, 2)
    compeps(func1d(x, degree), func1d(x, degree, 2, f2, f1), 1.E-10)
end

function testRecursiveTchebychevSecondDerivative(degree::Integer, x::Real)
    @assert degree >= 2 "degree must be >= 2"
    prime1 = VectorSpaceLeastSquares.dtchebychev1d(x, 1)
    prime2 = VectorSpaceLeastSquares.dtchebychev1d(x, 2)
    second1 = VectorSpaceLeastSquares.d2tchebychev1d(x, 1)
    second2 = VectorSpaceLeastSquares.d2tchebychev1d(x, 2)
    compeps(VectorSpaceLeastSquares.d2tchebychev1d(x, degree), VectorSpaceLeastSquares.d2tchebychev1d(x, degree, 2, prime2, prime1, second2, second1), 1.E-10)
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
    @test testRecursiveDerivative(4, 3., VectorSpaceLeastSquares.hermite1d)
    @test testRecursiveDerivative(2, 3., VectorSpaceLeastSquares.hermite1d)
    @test testRecursiveDerivative(6, 3., VectorSpaceLeastSquares.hermite1d)
    @test testRecursiveDerivative(4, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testRecursiveDerivative(2, 3., VectorSpaceLeastSquares.tchebychev1d)
    @test testRecursiveDerivative(6, 3., VectorSpaceLeastSquares.tchebychev1d)
end

@testset "Differentiate 1d Tchebychev polynomials" begin
    @test testRecursiveDerivative(2, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testRecursiveDerivative(3, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testRecursiveDerivative(4, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testRecursiveDerivative(5, 3., VectorSpaceLeastSquares.dtchebychev1d)
    @test testRecursiveDerivative(6, 3., VectorSpaceLeastSquares.dtchebychev1d)
end

@testset "Differentiate 1d Tchebychev polynomials twice" begin
   @test testRecursiveTchebychevSecondDerivative(3, 2.5)
   @test testRecursiveTchebychevSecondDerivative(4, 2.5)
   @test testRecursiveTchebychevSecondDerivative(5, 2.5)
   @test testRecursiveTchebychevSecondDerivative(6, 2.5)
   @test testRecursiveTchebychevSecondDerivative(7, 2.5)
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

function testFitVoidTransformationPolynomialBasis(polType::PolynomialType, T::Type, eps)
    dim = 4
    deg = 3
    nSamples = 10000
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    df(x) = [
        - x[4] + 7 * x[3]^2,
        6 * x[2]^2,
        14 * x[3]^1 * x[1],
        - x[1]
    ]
    d2f(x) = [
        0.  0.  14 * x[3]  -1.
        0.  12 * x[2]  0.  0.
        14 * x[3]  0.  14 * x[1]  0.
        -1.  0.  0.  0.
    ]
    data = [randn(T, dim) for i in 1:nSamples]
    y = f.(data)
    vslsq = VSLeastSquares(PolynomialBasis(polType, dim, deg), VoidTransformation(), T)
    fit(vslsq, data, y)
    x = randn(T, dim)
    @test compeps(predict(vslsq, x), f(x), T(eps))
    @test compeps(gradient(vslsq, x), df(x), T(eps))
    @test compeps(hessian(vslsq, x), d2f(x), T(eps))
end

function testFitLinearTransformationPolynomialBasis(polType::PolynomialType, T::Type, eps)
    dim = 4
    deg = 3
    nSamples = 10000
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    df(x) = [
        - x[4] + 7 * x[3]^2
        6 * x[2]^2
        14 * x[3] * x[1]
        - x[1]
    ]
    d2f(x) = [
        0.  0.  14 * x[3]  -1.
        0.  12 * x[2]  0.  0.
        14 * x[3]  0.  14 * x[1]  0.
        -1.  0.  0.  0.
    ]
    data = [1.0 .+ 2.0 .* randn(T, dim) for i in 1:nSamples]
    y = f.(data)
    transformation = LinearTransformation(data)
    vslsq = VSLeastSquares(PolynomialBasis(polType, dim, deg), transformation, T)
    fit(vslsq, data, y)
    x = randn(T, dim)
    @test compeps(predict(vslsq, x), f(x), T(eps))
    @test compeps(gradient(vslsq, x), df(x), T(eps))
    @test compeps(hessian(vslsq, x), d2f(x), T(eps))
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

function testRidgePolynomialBasis(T::Type, eps)
    dim = 4
    deg = 5
    nSamples = 100
    f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
    data = [randn(T, dim) for i in 1:nSamples]
    y = f.(data)
    vslsq = VSLeastSquares(PolynomialBasis(Hermite, dim, deg), VoidTransformation(), T)
    alpha = LinRange(0., 0.1, 100)
    x = randn(T, dim)
    for a in alpha
        println(a)
        fit(vslsq, data, y, T(a))
        compeps(predict(vslsq, x), f(x), T(eps))
    end
    return compeps(predict(vslsq, x), f(x), T(eps))
end

function testRidgePiecewiseConstantBasis(T::Type, dim, eps)
    nIntervals = 50
    nSamples = 1000
    data = [rand(T, dim) for i in 1:nSamples]
    f(x) = log(1. + sum(x.^2))
    y = f.(data)
    vslsq = VSLeastSquares(PiecewiseConstantBasis(dim, nIntervals), VoidTransformation(), T)
    fit(vslsq, data, y, T(0.01))
    x = rand(T, dim)
    @test compeps(predict(vslsq, x), f(x), T(eps))
end

@testset "Least squares void transformation with polynomials " begin
    testFitVoidTransformationPolynomialBasis(Canonic,Float32, 1.E-3)
    testFitVoidTransformationPolynomialBasis(Canonic, Float64, 1.E-10)
    testFitVoidTransformationPolynomialBasis(Hermite,Float32, 1.E-3)
    testFitVoidTransformationPolynomialBasis(Hermite, Float64, 1.E-10)
    testFitVoidTransformationPolynomialBasis(Tchebychev,Float32, 1.E-3)
    testFitVoidTransformationPolynomialBasis(Tchebychev, Float64, 1.E-10)
end

@testset "Least squares void transformation with piecewise constant basis" begin
    testVoidTransformationPiecewiseConstantBasis(Float64, 1, 1.E-2)
    testVoidTransformationPiecewiseConstantBasis(Float64, 2, 1.E-2)
end

@testset "Least squares linear transformation with polynomials" begin
    testFitLinearTransformationPolynomialBasis(Canonic, Float64, 1.E-3)
    testFitLinearTransformationPolynomialBasis(Hermite, Float64, 1.E-3)
    testFitLinearTransformationPolynomialBasis(Tchebychev, Float64, 1.E-3)
end

"""
Black-Scholes price
"""
function bsprice(t::Real, spot::Real, sigma::Real, r::Real, T::Real, K::Real)
    timeToMaturity = T - t
    if (timeToMaturity <= 0.0) || (sigma <= 0.0)
        return max(spot - K, 0.)
    end
    d1 = (log(spot / K) + (r + sigma * sigma / 2) * timeToMaturity) / (sigma * sqrt(timeToMaturity))
    d2 = d1 - sigma * sqrt(timeToMaturity)
    return spot * cdf(Normal(), d1) - K * exp(-r * timeToMaturity) * cdf(Normal(), d2)
end

"""
Black-Scholes delta
"""
function bsdelta(t::Real, spot::Real, sigma::Real, r::Real, T::Real, K::Real)
    timeToMaturity = T - t
    if (timeToMaturity <= 0.0) || (sigma <= 0.0)
        return max(spot - K, 0.)
    end
    d1 = (log(spot / K) + (r + sigma * sigma / 2) * timeToMaturity) / (sigma * sqrt(timeToMaturity))
    return cdf(Normal(), d1)
end

"""
Black-Scholes gamma
"""
function bsgamma(t::Real, spot::Real, sigma::Real, r::Real, T::Real, K::Real)
    timeToMaturity = T - t
    if (timeToMaturity <= 0.0) || (sigma <= 0.0)
        return max(spot - K, 0.)
    end
    d1 = (log(spot / K) + (r + sigma * sigma / 2) * timeToMaturity) / (sigma * sqrt(timeToMaturity))
    return 1. / (spot * sigma * sqrt(timeToMaturity)) * pdf(Normal(), d1)
end

function testGPR4BS()
    S0 = 100
    K = 100
    sigma = 0.2
    T = 1
    r = 0.03
    spaceGrid = range(50, 200; length=50)
    prices = bsprice.(0., spaceGrid, sigma, r, T, K)
    
    gaussianKernel = GaussianKernel(8.)
    kernelBasis = KernelBasis(Float64, gaussianKernel, 1)
    vslsq = VSLeastSquares(kernelBasis, VoidTransformation(), Float64)
    fit(vslsq, [[x] for x in spaceGrid], prices)

    spaceGridTest = range(60, 150; length=500)
    precision = 0.001
    predictedPrices = predict.(vslsq, [[x] for x in spaceGridTest])
    @test compeps(predictedPrices, bsprice.(0., spaceGridTest, sigma, r, T, K), precision)
    predictedDeltas = derivative.(vslsq, [[x] for x in spaceGridTest], 1)
    @test compeps(predictedDeltas, bsdelta.(0., spaceGridTest, sigma, r, T, K), precision)
    predictedGamma = secondDerivative.(vslsq, [[x] for x in spaceGridTest], 1, 1)
    @test compeps(predictedGamma, bsgamma.(0., spaceGridTest, sigma, r, T, K), precision)
end

@testset "GPR via LS regression" begin
    testGPR4BS()
end