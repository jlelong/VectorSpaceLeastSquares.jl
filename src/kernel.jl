# Kernel basis

"""
    AbstractKernel

Abstract kernel structure. A `Kernel` must implement [`kernel`](@ref) and [`dkernel`](@ref)
"""
abstract type AbstractKernel end

"""
    KernelBasis{Tk, Td} <: AbstractBasis where {Tk <: AbstractKernel, Td <: Real}

Kernel basis
"""
struct KernelBasis{Tk, Td} <: AbstractBasis where {Tk <: AbstractKernel, Td <: Real}
    nVariates::Int64
    nodes::Vector{Vector{Td}}
    kernel::Tk
end

"""
    KernelBasis(kernel::Tk, nodes::Vector{Vector{Td}}) where {Tk <: AbstractKernel, Td <: Real}

"""
function KernelBasis(kernel::Tk, typeTd::Type{Td}, nVariates::Integer) where {Tk <: AbstractKernel, Td <: Real}
    nodes = Vector{Vector{typeTd}}(undef, 0)
    KernelBasis(nVariates, nodes, kernel)
end

nVariates(p::KernelBasis) = p.nVariates
length(p::KernelBasis) = length(p.nodes)
getType(p::KernelBasis) = p.type
isDifferentiable(p::KernelBasis) = true


"""
    kernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Compute the value at `x` of the kernel centered at `node`.
"""
function kernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end

"""
    dkernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)

Compute the first derivative w.r.t the `derivativeIndex` coordinate at `x` of the kernel centered at `node`.
"""
function dkernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer) end


struct GaussianKernel <: AbstractKernel 
    sigmaSq::Real
end

"""
    kernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Compute the value at `x` of the Gaussian kernel centered at `node`.
"""
function kernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return exp(-0.5 * sum((node .- x).^2 / k.sigmaSq)) / (2 * pi * k.sigmaSq)^(0.5 * length(node))
end

"""
    dkernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)

Compute the first derivative w.r.t the `derivativeIndex` coordinate at `x` of the Gaussian kernel centered at `node`.
"""
function dkernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)
    return - (x[derivativeIndex] - node[derivativeIndex]) / k.sigmaSq * kernel(k, node, x)
end

"""
    d2kernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex1::Integer, derivativeIndex2::Integer)

Compute the second derivative w.r.t the (`derivativeIndex1`,`derivativeIndex2`)  coordinates at `x` of the Gaussian kernel centered at `node`.
"""
function d2kernel(k::GaussianKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex1::Integer, derivativeIndex2::Integer)
    term1 = (x[derivativeIndex1] - node[derivativeIndex1]) * (x[derivativeIndex2] - node[derivativeIndex2]) / k.sigmaSq^2 * kernel(k, node, x)
    if derivativeIndex2 == derivativeIndex1
        term1 += - 1. / k.sigmaSq * kernel(k, node, x)
    end
    return term1
end


"""
    value(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer)

Compute the value of the kernel centered around the `index`-th node at point `x`.
"""
function value(b::KernelBasis{Tk, Td}, x::AbstractVector{Td}, index::Integer) where {Tk <: AbstractKernel, Td <: Real}
    return kernel(b.kernel, b.nodes[index], x)
end

"""
    derivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)

Compute the value of the first derivative of the kernel centered around the `index`-th node w.r.t to the `derivativeIndex` coordinate at point `x`.
"""
function derivative(b::KernelBasis{Tk, Td}, x::AbstractVector{Td}, index::Integer, derivativeIndex::Integer)  where {Tk <: AbstractKernel, Td <: Real}
    return dkernel(b.kernel, b.nodes[index], x, derivativeIndex)
end

"""
    secondDerivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, derivativeIndex1::Integer, derivativeIndex2::Integer)

Compute the value of the first derivative of the `index`-th basis function w.r.t to the `derivativeIndex` variate at point `x`.
"""
function secondDerivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, derivativeIndex1::Integer, derivativeIndex2::Integer)
    return d2kernel(b.kernel, b.nodes[index], x, derivativeIndex1, derivativeIndex2)
end
