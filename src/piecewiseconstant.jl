# Piecewise constant local basis

struct PiecewiseConstant <: AbstractBasis
    nVariates::Integer
    nIntervals::Vector{<:Integer}
    size::Integer
end

function PiecewiseConstant(nVariates::Integer, nIntervals::Integer)
    nIntervalsVect = Vector{Integer}(undef, nVariates)
    nIntervalsVect .= nIntervals
    dim = prod(nIntervals)
    return PiecewiseConstant(nVariates, nIntervalsVect, dim)
end

nVariates(p::PiecewiseConstant) = p.nVariates
length(p::PiecewiseConstant) = p.size
