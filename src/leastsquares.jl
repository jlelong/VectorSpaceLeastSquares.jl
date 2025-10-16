abstract type AbstractTransform end

struct LinearTransform <: AbstractTransform
    scale::Vector{<:Real}
    center::Vector{<:Real}
end

"""
Create a linear transformation by setting `center` as the empirical mean and `scale` as the inverse of the empirical standard deviation. Each entry of `x` is supposed one sample of the data
"""
function LinearTransform(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
    dim = 0
    nSamples = length(x)
    if nSamples >= 1
        dim = length(x[1])
    end
    center = Vector{T}(undef, dim)
    squares = Vector{T}(undef, dim)
    center .= 0
    squares .= 0
    for xi in x
        for j in 1:dim
            center[j] += xi[j]
            squares[j] += xi[j] * xi[j]
        end
    end
    center .= center ./ nSamples
    squares .= (squares ./ nSamples .- center.^2)
    LinearTransform(squares, center)
end

abstract type NonLinearTransform <: AbstractTransform end

struct VSLeastSquares{T<:Real}
    basis::AbstractBasis
    transform::AbstractTransform
    coefficients::Vector{T}
end

function VSLeastSquares{T}(basis::AbstractBasis, transform::AbstractTransform) where T<:Real
    coefficients = Vector{T}(undef, size(basis))
    VSLeastSquares{T}(basis, transform, coefficients)
end


