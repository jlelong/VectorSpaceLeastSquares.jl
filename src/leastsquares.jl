using LinearAlgebra.BLAS: ger!
abstract type AbstractTransformation end
struct VoidTransformation <: AbstractTransformation end

"""
Apply the transformation `t` to `x` ans store the result in `tx`
"""
function apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end

"""
Compute `\\partial_{x_j} t_i(x)`
"""
function jacobian(t::AbstractTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer) end

function apply!(t::VoidTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    tx .= x
end

function jacobian(t::VoidTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer)
    return Int(i == j)
end

struct LinearTransformation{Td} <: AbstractTransformation where Td<:Real
    scale::Vector{Td}
    center::Vector{Td}
end

function apply!(t::LinearTransformation{Td}, tx::AbstractVector{Td}, x::AbstractVector{Td}) where Td<:Real
    tx .= (x .- t.center) .* t.scale
end

function jacobian(t::LinearTransformation{Td}, x::AbstractVector{Td}, i::Integer, j::Integer) where Td<:Real
    if i != j
        return 0.
    else
        return t.scale[i]
    end
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
    squares .= 1. ./ sqrt.((squares ./ nSamples .- center.^2))
    LinearTransformation(squares, center)
end

abstract type NonLinearTransform <: AbstractTransformation end

struct VSLeastSquares{Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    basis::Tb
    transformation::Tt
    coefficients::Vector{Td}
    _transformed_data::Vector{Td}
end

"""
Create a VSLeastSquares object from `basis` and `transform` and capable of handling `Td` typed data.
"""
function VSLeastSquares(basis::Tb, transform::Tt=VoidTransformation(), Td::Type=Float64) where {Tb<:AbstractBasis, Tt<:AbstractTransformation}
    coefficients = Vector{Td}(undef, length(basis))
    transformed_data = Vector{Td}(undef, nVariates(basis))
    VSLeastSquares{Tb, Tt, Td}(basis, transform, coefficients, transformed_data)
end

length(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = Int64(length(vslsq.basis))
getCoefficients(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.coefficients
getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.basis

"""
Solve the least squares problem
"""
function fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    nSamples = length(x)
    A = Matrix{Td}(undef, length(vslsq), length(vslsq))
    b = Vector{Td}(undef, length(vslsq))
    phi_k = Vector{Td}(undef, length(vslsq))
    A .= 0.
    b .= 0.
    phi_k .= 0
    for i in 1:nSamples
        apply!(vslsq.transformation, vslsq._transformed_data, x[i])
        for k in 1:length(vslsq)
            phi_k[k] = value(vslsq.basis, vslsq._transformed_data, k)
            b[k] += phi_k[k] * y[i]
        end
        ger!(Td(1.), phi_k, phi_k, A)
    end
    vslsq.coefficients .= A \ b
end

"""
Compute the value predicted by the least squares problem.

The method `fit` must have been called before.
"""
function predict(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, vslsq._transformed_data, x)
    for i in 1:length(vslsq)
        v = value(basis, vslsq._transformed_data, i)
        c = coefficients[i]
        val += c * v
    end
    return val
end

"""
Compute the partial derivative of the prediction w.r.t to the `index` variable

The method `fit` must have been called before.
"""
function derivative(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, vslsq._transformed_data, x)
    for i in 1:length(vslsq)
        di = 0.
        for j in 1:length(x)
            dval = derivative(basis, vslsq._transformed_data, i, j)
            dphi = jacobian(vslsq.transformation, x, j, index)
            di += dval * dphi
        end
        c = coefficients[i]
        val += c * di
    end
    return val
end

"""
Compute the gradient of the prediction

The method `fit` must have been called before.
"""
function gradient(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    return [derivative(vslsq, x, i) for i in 1:length(x)]
end