module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export Polynomial, PiecewiseConstant, AbstractBasis
export value, derivative, nVariates, size
export LinearTransformation, AbstractTransformation, VoidTransformation, VSLeastSquares, fit, predict, getCoefficients
end
