module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export PolynomialBasis, PiecewiseConstant, AbstractBasis
export value, derivative, nVariates, size, tensor, type, basis
export LinearTransformation, AbstractTransformation, VoidTransformation, VSLeastSquares, fit, predict, derivative, gradient, getCoefficients
end
