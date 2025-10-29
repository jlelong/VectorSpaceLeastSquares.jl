using SparseArrays
import Base: length, size


"""
    AbstractBasis

Super type for all bases. A basis must implement the following methods: [`nVariates`](@ref), [`length`](@ref), [`getType`](@ref), [`value`](@ref). If the basis contains only differentiable functions, it must also implement [`derivative`](@ref). If the method `derivative` is not defined for a concrete subtype of `AbstractBasis`, it means that the basis functions are NOT differentiable.
"""
abstract type AbstractBasis end

"""
    nVariates(B::AbstractBasis)

Return the number of variates of the functions inside the basis.
"""
function nVariates(B::AbstractBasis) end

"""
    length(B::AbstractBasis)

Return the number of elements in the basis.
"""
function length(B::AbstractBasis) end

"""
    size(B::AbstractBasis)

Return the tuple ([`nVariates`](@ref), [`length`](@ref)).
"""
size(B::AbstractBasis) = (nVariates(B), length(B))


"""
    getType(B::AbstractBasis)

Return the internal basis type.
"""
function getType(B::AbstractBasis) end

"""
    value(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer)

Compute the value of the `index`-th basis function at point `x`.
"""
function value(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer) end

"""
    derivative(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)

Compute the value of the first derivative of the `index`-th basis function w.r.t to the `derivativeIndex` variate at point `x`.
"""
function derivative(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)
    error("Basis is not differentiable $getType(B).")
end

include("polynomial.jl")
include("piecewiseconstant.jl")
