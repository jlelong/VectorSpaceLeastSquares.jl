using LinearAlgebra: I
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
Base.broadcastable(x::VSLeastSquares) = Ref(x)

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
    getTx(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Return the vector internally used to store the transformed data. Internal use only.
"""
getTx(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = vslsq._transformed_data

"""
    fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Solve the least squares problem.
"""
function fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
    nSamples = length(x)
    A = zeros(Td, length(vslsq), length(vslsq))
    b = zeros(Td, length(vslsq))
    phi_k = zeros(Td, length(vslsq))
    for i in 1:nSamples
        apply!(vslsq.transformation, getTx(vslsq), x[i])
        for k in 1:length(vslsq)
            phi_k[k] = value(vslsq.basis, getTx(vslsq), k)
            b[k] += phi_k[k] * y[i]
        end
        ger!(Td(1.), phi_k, phi_k, A)
    end
    vslsq.coefficients .= (A + lambda * nSamples * I) \ b
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
    if isa(vslsq.transformation, VoidTransformation)
        tx = x
    else
        tx = getTx(vslsq)
        apply!(vslsq.transformation, tx, x)
    end
    for i in 1:length(vslsq)
        v = value(basis, tx, i)
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
    apply!(vslsq.transformation, getTx(vslsq), x)
    for i in 1:length(vslsq)
        di = 0.
        for j in 1:length(x)
            dval = derivative(basis, getTx(vslsq), i, j)
            dphi = jacobian(vslsq.transformation, x, j, index)
            di += dval * dphi
        end
        c = coefficients[i]
        val += c * di
    end
    return val
end

"""
    derivative(vslsq::VSLeastSquares{Tb, VoidTransformation, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Td<:Real}

Compute the partial derivative of the prediction w.r.t to the `index` variable for a [`VoidTransformation`](@ref).

The method [`fit`](@ref) must have been called before.
"""
function derivative(vslsq::VSLeastSquares{Tb, VoidTransformation, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Td<:Real}
    @assert isDifferentiable(getBasis(vslsq)) "The basis must be differentiable to call `derivative`."
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    for i in 1:length(vslsq)
        di = derivative(basis, x, i, index)
        c = coefficients[i]
        val += c * di
    end
    return val
end

"""
    derivative(vslsq::VSLeastSquares{Tb, LinearTransformation, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Td<:Real}

Compute the partial derivative of the prediction w.r.t to the `index` variable for a linear transformation.

The method [`fit`](@ref) must have been called before.
"""
function derivative(vslsq::VSLeastSquares{Tb, LinearTransformation{Td}, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Td<:Real}
    @assert isDifferentiable(getBasis(vslsq)) "The basis must be differentiable to call `derivative`."
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, getTx(vslsq), x)
    for i in 1:length(vslsq)
        di = derivative(basis, getTx(vslsq), i, index) * vslsq.transformation.scale[index]
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

"""
    secondDerivative(vslsq::VSLeastSquares{Tb, VoidTransformation, Td}, x::AbstractVector{Td}, i::Integer, j::Integer) where {Tb<:AbstractBasis, Td<:Real}

Compute the second order partial derivative of the prediction w.r.t to the `(i,j)` variables for a void transformation.

The method [`fit`](@ref) must have been called before.
"""
function secondDerivative(vslsq::VSLeastSquares{Tb, VoidTransformation, Td}, x::AbstractVector{Td}, index1::Integer, index2::Integer) where {Tb<:AbstractBasis, Td<:Real}
    @assert isTwiceDifferentiable(getBasis(vslsq)) "The basis must be twice differentiable to call `derivative`."
    val = 0.
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    for i in 1:length(vslsq)
        c = coefficients[i]
        ddi = secondDerivative(basis, x, i, index1, index2)
        val += c * ddi
    end
    return val
end

"""
    secondDerivative(vslsq::VSLeastSquares{Tb, LinearTransformation, Td}, x::AbstractVector{Td}, i::Integer, j::Integer) where {Tb<:AbstractBasis, Td<:Real}

Compute the second order partial derivative of the prediction w.r.t to the `(i,j)` variables for a linear transformation.

The method [`fit`](@ref) must have been called before.
"""
function secondDerivative(vslsq::VSLeastSquares{Tb, LinearTransformation{Td}, Td}, x::AbstractVector{Td}, index1::Integer, index2::Integer) where {Tb<:AbstractBasis, Td<:Real}
    @assert isTwiceDifferentiable(getBasis(vslsq)) "The basis must be twice differentiable to call `derivative`."
    val = 0.
    coefficients = getCoefficients(vslsq)
    apply!(vslsq.transformation, getTx(vslsq), x)
    basis = getBasis(vslsq)
    for i in 1:length(vslsq)
        c = coefficients[i]
        ddi = secondDerivative(basis, getTx(vslsq), i, index1, index2) * vslsq.transformation.scale[index1] * vslsq.transformation.scale[index2]
        val += c * ddi
    end
    return val
end

"""
    hessian(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Compute the hessian of the prediction at `x`. It only works if the transformation `Tt` is a void or a linear transformation.

The method [`fit`](@ref) must have been called before.
"""
hessian(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real} = [secondDerivative(vslsq, x, i, j) for i in 1:length(x), j in 1:length(x)]


#
# Specific methods for PiecewiseConstantBasis
#

"""
    fit(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tt<:AbstractTransformation, Td<:Real}

Solve the least squares problem using the specific structure of the [`PiecewiseConstantBasis`](@ref).
"""
function fit(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tt<:AbstractTransformation, Td<:Real}
    nSamples = length(x)
    count = zeros(Int64, length(vslsq))
    coefficients = getCoefficients(vslsq)
    coefficients .= 0
    for i in 1:nSamples
        apply!(vslsq.transformation, getTx(vslsq), x[i])
        globalIndex = computeGlobalIndex(getBasis(vslsq), getTx(vslsq))
        if globalIndex != -1
            count[globalIndex] += 1
            vslsq.coefficients[globalIndex] += y[i]
        end
    end
    coefficients ./= (max.(count, 1) .+ nSamples * lambda)
end

"""
    predict(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}

Compute the value predicted by the least squares problem using the specific structure of the [`PiecewiseConstantBasis`](@ref).

The method [`fit`](@ref) must have been called before.
"""
function predict(vslsq::VSLeastSquares{PiecewiseConstantBasis, Tt, Td}, x::AbstractVector{Td}) where {Tt<:AbstractTransformation, Td<:Real}
    coefficients = getCoefficients(vslsq)
    basis = getBasis(vslsq)
    apply!(vslsq.transformation, getTx(vslsq), x)
    globalIndex = computeGlobalIndex(basis, getTx(vslsq))
    if globalIndex != -1
        return coefficients[globalIndex]
    else
        return 0.
    end
end


#
# Specific methods for KernelBasis
#

"""
    fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}

Solve the least squares problem.
"""
function fit(vslsq::VSLeastSquares{KernelBasis{Tk, Td}, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}, lambda::Td = Td(0)) where {Tk<:AbstractKernel, Tt<:AbstractTransformation, Td<:Real}
    @assert length(x) == length(y) "Size mismatch in fit"
    nSamples = length(x)
    resize!(vslsq.basis.nodes, length(x))
    resize!(vslsq.coefficients, length(x))
    # Create the nodes of the kernels by applying the transformation
    for i in 1:nSamples
        vslsq.basis.nodes[i] = Vector{Td}(undef, nVariates(vslsq.basis))
        apply!(vslsq.transformation, vslsq.basis.nodes[i], x[i])
    end

    # Solve the least squares problem
    K = Matrix{Td}(undef, nSamples, nSamples)
    for i in 1:nSamples
        for j in 1:nSamples
            K[i,j] = kernel(vslsq.basis.kernel, vslsq.basis.nodes[i], vslsq.basis.nodes[j])
        end
    end
    vslsq.coefficients .= (K + lambda * I) \ y
end
