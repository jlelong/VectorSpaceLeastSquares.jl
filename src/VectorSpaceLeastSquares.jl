module VectorSpaceLeastSquares

# Write your package code here.
include("basis.jl")
include("leastsquares.jl")

export PolynomialType, Canonic, Hermite, Tchebychev
export Polynomial, PiecewiseConstant, LinearTransform, AbstractBasis
export value, derivative, nVariates, size
export LinearTransform, AbstractTransform, VSLeastSquares
end
