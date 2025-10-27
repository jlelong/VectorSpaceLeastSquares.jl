using LinearAlgebra.BLAS: ger!

"""
    AbstractTransformation

Super type for all transformations. 

A transformation is a function ``\\varphi: \\mathbb{R}^d \\to \\mathbb{R}^d``, which is applied on the fly to the data before proceeding with the least squares problem. A transformation must implement [`apply!`](@ref) and [`jacobian`](@ref)
"""
abstract type AbstractTransformation end

"""
    VoidTransformation <: AbstractTransformation

This transformation does nothing
"""
struct VoidTransformation <: AbstractTransformation end

"""
    apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Apply the transformation `t` to `x` and store the result in `tx`
"""
function apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end

"""
    jacobian(t::AbstractTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer)

Compute ``\\partial_{x_j} \\varphi_i(x)``
"""
function jacobian(t::AbstractTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer) end

function apply!(t::VoidTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    tx .= x
end

function jacobian(t::VoidTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer)
    return Int(i == j)
end

"""
    LinearTransformation{Td} <: AbstractTransformation where Td<:Real

Implement a linear transformation of the data defined by ``\\varphi(x) = (x - \\alpha) * \\sigma`` 
"""
struct LinearTransformation{Td} <: AbstractTransformation where Td<:Real
    scale::Vector{Td}
    center::Vector{Td}
end

"""
    getCenter(t::LinearTransformation{<:Real})

Return the center α of the linear transformation
"""
getCenter(t::LinearTransformation{<:Real}) = t.center

"""
    getScale(t::LinearTransformation{<:Real}) = t.scale

Return the scale σ of the linear transformation
"""
getScale(t::LinearTransformation{<:Real}) = t.scale

"""
    LinearTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real

Create a linear transformation by setting α as the empirical mean and σ as the inverse of the empirical standard deviation. Each entry of `x` is supposed to be one sample of the data.
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
    VSLeastSquares{Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

The main object to solve a least squares problem.
"""
struct VSLeastSquares{Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    basis::Tb
    transformation::Tt
    coefficients::Vector{Td}
    _transformed_data::Vector{Td}
end

"""
    VSLeastSquares(basis::Tb, transform::Tt=VoidTransformation(), Td::Type=Float64) where {Tb<:AbstractBasis, Tt<:AbstractTransformation}

Create a VSLeastSquares object from `basis` and `transform` and capable of handling `Td` typed data.
"""
function VSLeastSquares(basis::Tb, transform::Tt=VoidTransformation(), Td::Type=Float64) where {Tb<:AbstractBasis, Tt<:AbstractTransformation}
    coefficients = Vector{Td}(undef, length(basis))
    transformed_data = Vector{Td}(undef, nVariates(basis))
    VSLeastSquares{Tb, Tt, Td}(basis, transform, coefficients, transformed_data)
end


"""
    length(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the number of functions of the basis used to solve the least squares problem
"""
length(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = length(vslsq.basis)

"""
    nVariates(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the number of variates in the least squares problem
"""
nVariates(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = nVariates(vslsq.basis)

"""
    size(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the tuple (nVariates, length)
"""
size(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = size(vslsq.basis)

"""
    getCoefficients(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the coefficients solution to the least squares problem. The function [`fit`](@ref) must have been called
"""
getCoefficients(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.coefficients

"""
    getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the basis used to solve the least squares problem.
"""
getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.basis


"""
    fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

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
    predict(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Compute the value predicted by the least squares problem.

The method [`fit`](@ref) must have been called before.
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
    derivative(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Compute the partial derivative of the prediction w.r.t to the `index` variable

The method [`fit`](@ref) must have been called before.
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
    gradient(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Compute the gradient of the prediction at `x`

The method [`fit`](@ref) must have been called before.
"""
gradient(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = [derivative(vslsq, x, i) for i in 1:length(x)]
