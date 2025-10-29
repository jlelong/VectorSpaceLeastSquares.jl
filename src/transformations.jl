"""
    AbstractTransformation

Super type for all transformations. 

A transformation is a function ``\\varphi: \\mathbb{R}^d \\to \\mathbb{R}^d``, which is applied on the fly to the data before proceeding with the least squares problem. A transformation must implement [`apply!`](@ref) and [`jacobian`](@ref).


To define a new transformation, define a new concrete subtype of `AbstractTransformation` and implement the corresponding [`apply!`](@ref) and [`jacobian`](@ref) methods. For instance the scaled log-transformation ``\\varphi(x) = \\alpha \\log(x)`` where ``\\alpha \\in \\mathbb{R}`` is defined as follows

```julia
struct LogTransformation <: AbstractTransformation
    scale::Vector{<:Real}
end

function apply!(t::LogTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    tx .= t.scale .* log.(x)
end

function jacobian(t::LogTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer)
    if i != j
        return 0.
    end
    t.scale[i] / x[i]
end
```
"""
abstract type AbstractTransformation end

"""
    VoidTransformation <: AbstractTransformation

This transformation does nothing.
"""
struct VoidTransformation <: AbstractTransformation end

"""
    apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Apply the transformation `t` to `x` and store the result in `tx`.
"""
function apply!(t::AbstractTransformation, tx::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end

"""
    jacobian(t::AbstractTransformation, x::AbstractVector{<:Real}, i::Integer, j::Integer)

Compute ``\\partial_{x_j} \\varphi_i(x)``.
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

