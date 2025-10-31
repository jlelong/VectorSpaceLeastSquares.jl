using LinearAlgebra.BLAS: ger!

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

Return the coefficients solution to the least squares problem. 
"""
getCoefficients(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.coefficients

"""
    getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the basis used to solve the least squares problem.
"""
getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq.basis


"""
    fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Solve the least squares problem.
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

Compute the partial derivative of the prediction w.r.t to the `index` variable.

The method [`fit`](@ref) must have been called before.
"""
function derivative(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    @assert isDifferentiable(getBasis(vslsq)) "The basis must be differentiable to call `derivative`."
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

Compute the gradient of the prediction at `x`.

The method [`fit`](@ref) must have been called before.
"""
gradient(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = [derivative(vslsq, x, i) for i in 1:length(x)]


#
# Specific methods for PiecewiseConstantBasis
#

"""
    fit(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}

Solve the least squares problem using the specific structure of the [`PiecewiseConstantBasis`](@ref).
"""
function fit(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}
    nSamples = length(x)
    count = Vector{Int64}(undef, length(vslsq))
    count .= 0
    coefficients = getCoefficients(vslsq)
    coefficients .= 0
    for i in 1:nSamples
        apply!(vslsq.transformation, vslsq._transformed_data, x[i])
        globalIndex = computeGlobalIndex(getBasis(vslsq), vslsq._transformed_data)
        if globalIndex != -1
            count[globalIndex] += 1
            vslsq.coefficients[globalIndex] += y[i]
        end
    end
    coefficients ./= max.(count, 1)
end

"""
    predict(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}

Compute the value predicted by the least squares problem using the specific structure of the [`PiecewiseConstantBasis`](@ref).

The method [`fit`](@ref) must have been called before.
"""
function predict(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, vslsq._transformed_data, x)
    globalIndex = computeGlobalIndex(basis, vslsq._transformed_data)
    if globalIndex != -1
        return coefficients[globalIndex]
    else
        return 0.
    end
end