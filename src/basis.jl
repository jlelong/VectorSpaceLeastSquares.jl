using SparseArrays
import Base: length, size


"""
    AbstractBasis

Super type for all bases. A basis must implement the following methods: [`nVariates`](@ref), [`length`](@ref), [`getType`](@ref), [`isDifferentiable`](@ref), [`value`](@ref). If the basis contains only differentiable functions, in which case [`isDifferentiable`](@ref) returns `true`, it must also implement [`derivative`](@ref). If the basis contains only twice differentiable functions, in which case [`isTwiceDifferentiable`](@ref) returns `true`, it must also implement [`secondDerivative`](@ref).
"""
abstract type AbstractBasis end

"""
    nVariates(b::AbstractBasis)

Return the number of variates of the functions inside the basis.
"""
function nVariates(b::AbstractBasis) end

"""
    length(b::AbstractBasis)

Return the number of elements in the basis.
"""
function length(b::AbstractBasis) end

"""
    size(b::AbstractBasis)

Return the tuple ([`nVariates`](@ref), [`length`](@ref)).
"""
size(b::AbstractBasis) = (nVariates(B), length(B))


"""
    getType(b::AbstractBasis)

Return the internal basis type.
"""
function getType(b::AbstractBasis) end

"""
    value(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer)

Compute the value of the `index`-th basis function at point `x`.
"""
function value(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer) end

"""
    derivative(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, i::Integer)

Compute the value of the first derivative of the `index`-th basis function w.r.t to the `i` variate at point `x`. Only available if [`isDifferentiable`](@ref) returns true.
"""
function derivative(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)
    error("Basis is not differentiable $getType(B).")
end

"""
    secondDerivative(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, i1::Integer, i2::Integer)

Compute the value of the second derivative of the `index`-th basis function w.r.t to the `i1, i2` variates at point `x`. Only available if [`isTwiceDifferentiable`](@ref) returns true.
"""
function secondDerivative(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex1::Integer, derivativeIndex2::Integer)
    error("Basis is not differentiable $getType(B).")
end

"""
    isDifferentiable(b::AbstractBasis)

Return true if the functions in the basis are differentiable. In this case, a specific method [`derivative`](@ref) must be implemented.
"""
isDifferentiable(b::AbstractBasis) = false

"""
    isTwiceDifferentiable(b::AbstractBasis)

Return true if the functions in the basis are twice differentiable. In this case, a specific method [`secondDerivative`](@ref) must be implemented.
"""
isTwiceDifferentiable(b::AbstractBasis) = false

include("polynomial.jl")
include("piecewiseconstant.jl")
include("kernel.jl")
