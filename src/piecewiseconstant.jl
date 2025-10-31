# Piecewise constant local basis

"""
    PiecewiseConstantBasis

Represent a basis of local functions defined on ``[0,1]^d``

- `nVariates` is the dimension `d` of the space.
- `nIntervals` is a vector of size `d` defining the number of sub-intervals to use along each dimension
"""
struct PiecewiseConstantBasis <: AbstractBasis
    nVariates::Integer
    nIntervals::Vector{<:Integer}
    size::Integer
end

"""
    PiecewiseConstantBasis(nVariates::Integer, nIntervals::Integer)

Create a `PiecewiseConstantBasis` with `nIntervals` along each direction. 
"""
function PiecewiseConstantBasis(nVariates::Integer, nIntervals::Integer)
    nIntervalsVect = Vector{Integer}(undef, nVariates)
    nIntervalsVect .= nIntervals
    size = nIntervals^nVariates
    return PiecewiseConstantBasis(nVariates, nIntervalsVect, size)
end

"""
    PiecewiseConstantBasis(nVariates::Integer, nIntervals::Vector{<:Integer})

Create a `PiecewiseConstantBasis` by specifying the number of intervals per direction (possibly different to allow non squared grids).
"""
function PiecewiseConstantBasis(nVariates::Integer, nIntervals::Vector{<:Integer})
    size = prod(nIntervals)
    return PiecewiseConstantBasis(nVariates, nIntervalsVect, size)
end

nVariates(p::PiecewiseConstantBasis) = p.nVariates
length(p::PiecewiseConstantBasis) = p.size
getType(p::PiecewiseConstantBasis) = "PiecewiseConstantBasis"
isDifferentiable(p::PiecewiseConstantBasis) = false

"""
    computeGlobalIndex(p::PiecewiseConstantBasis, x::AbstractVector{<:Real})

Compute the linear index of grid cell in which `x` lies. Return -1 if `x` is not inside the grid.
"""
function computeGlobalIndex(p::PiecewiseConstantBasis, x::AbstractVector{<:Real})
    globalIndex = 1
    nIntervalsProd = 1
    for i in 1:length(x)
        dimIndex = floor(Int64, x[i] * p.nIntervals[i])
        if dimIndex < 0 || dimIndex >= p.nIntervals[i]
            return -1
        end
        globalIndex += dimIndex * nIntervalsProd;
        nIntervalsProd *= p.nIntervals[i]
    end
    return globalIndex
end

function value(p::PiecewiseConstantBasis, x::AbstractVector{<:Real}, index::Integer)
    globalIndex = computeGlobalIndex(p, x)
    if globalIndex == index
        return 1
    else
        return 0
    end
end