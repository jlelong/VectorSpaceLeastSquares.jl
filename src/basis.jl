using SparseArrays
import Base: size

abstract type AbstractBasis end
function nVariates(::AbstractBasis) end
function size(::AbstractBasis) end

struct LinearTransform
    scale::Vector{<:Real}
    center::Vector{<:Real}
end

include("polynomial.jl")
include("piecewiseconstant.jl")
