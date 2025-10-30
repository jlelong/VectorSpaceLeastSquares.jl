module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("transformations.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export PolynomialBasis, PiecewiseConstantBasis, AbstractBasis
export value, derivative, nVariates, length, size, getTensor, getType
export AbstractTransformation, VoidTransformation, apply!, jacobian
export LinearTransformation, getCenter, getScale
export VSLeastSquares, fit, predict, derivative, gradient, getCoefficients, getBasis
end
