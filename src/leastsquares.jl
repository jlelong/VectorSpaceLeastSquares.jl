using LinearAlgebra.BLAS: ger!
abstract type AbstractTransformation end
struct VoidTransformation <: AbstractTransformation end

function apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end
function apply!(t::VoidTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real}) 
    tx .= x
end

struct LinearTransformation <: AbstractTransformation
    scale::Vector{<:Real}
    center::Vector{<:Real}
end

function apply!(t::LinearTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    tx .= (x .- t.center) .* scale
end


"""
Create a void linear transformation: `center `= 0 and `scale` = 1
"""
function LinearTransformation(dim::Integer, T::Type=Float64)
    center = Vector{T}(undef, dim)
    scale = Vector{T}(undef, dim)
    center .= 0
    scale .= 1
    LinearTransformation(scale, center)
end

"""
Create a linear transformation by setting `center` as the empirical mean and `scale` as the inverse of the empirical standard deviation. Each entry of `x` is supposed one sample of the data
"""
function LinearTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
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
    LinearTransformation(squares, center)
end

abstract type NonLinearTransform <: AbstractTransformation end

struct VSLeastSquares{T<:Real}
    basis::AbstractBasis
    transformation::AbstractTransformation
    coefficients::Vector{T}
    _transformed_data::Vector{<:Real}
end

function VSLeastSquares{T}(basis::AbstractBasis, transform::AbstractTransformation) where T<:Real
    coefficients = Vector{T}(undef, size(basis))
    transformed_data = Vector{T}(undef, nVariates(basis))
    VSLeastSquares{T}(basis, transform, coefficients, transformed_data)
end

VSLeastSquares{T}(basis::AbstractBasis) where T<:Real = VSLeastSquares{T}(basis, VoidTransformation())


size(vslsq::VSLeastSquares) = size(vslsq.basis)
getCoefficients(vslsq::VSLeastSquares) = vslsq.coefficients
getBasis(vslsq::VSLeastSquares) = vslsq.basis

function fit(vslsq::VSLeastSquares{T}, x::AbstractVector{<:AbstractVector{T}}, y::AbstractVector{T}) where T<:Real
    nSamples = length(x)
    A = Matrix{T}(undef, size(vslsq), size(vslsq))
    b = Vector{T}(undef, size(vslsq))
    phi_k = Vector{T}(undef, size(vslsq))
    A .= 0.
    b .= 0.
    phi_k .= 0
    for i in 1:nSamples
        apply!(vslsq.transformation, vslsq._transformed_data, x[i])
        for k in 1:size(vslsq)
            phi_k[k] = value(vslsq.basis, vslsq._transformed_data, k)
            b[k] += phi_k[k] * y[i]
        end
        ger!(1., phi_k, phi_k, A)
    end
    vslsq.coefficients .= A \ b
end

function predict(vslsq::VSLeastSquares{T}, x::AbstractVector{T}) where T<:Real
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, vslsq._transformed_data, x)
    for i in 1:size(vslsq)
        val += coefficients[i] * value(basis, vslsq._transformed_data, i)
    end
    return val
end