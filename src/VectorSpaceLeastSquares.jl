module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("transformations.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export PolynomialBasis, PiecewiseConstantBasis, AbstractBasis
export value, derivative, secondDerivative, nVariates, length, size, getTensor, getType, isDifferentiable, isTwiceDifferentiable
export AbstractTransformation, VoidTransformation, apply!, jacobian
export LinearTransformation, getCenter, getScale
export GaussianTransformation, LogNormalTransformation, getMean, getSigma
export VSLeastSquares, fit, predict, derivative, gradient, hessian, getCoefficients, getBasis
export AbstractKernel, KernelBasis, GaussianKernel, kernel, dkernel, d2kernel
end
