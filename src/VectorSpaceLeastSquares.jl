module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export PolynomialBasis, PiecewiseConstant, AbstractBasis
export value, derivative, nVariates, length, getTensor, getType
export LinearTransformation, AbstractTransformation, VoidTransformation, VSLeastSquares, fit, predict, derivative, gradient, getCoefficients, getBasis
end
