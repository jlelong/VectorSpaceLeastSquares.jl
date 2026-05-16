# Kernel basis

"""
    AbstractKernel

Abstract kernel structure. A  concrete `kernel` structure must implement [`kernel`](@ref) and [`dkernel`](@ref).
"""
abstract type AbstractKernel end

"""
    KernelBasis{Tk, Td} <: AbstractBasis where {Tk <: AbstractKernel, Td <: Real}

Represent a kernel basis where `Tk` must be a concrete implementation of an `AbstractKernel`.
"""
struct KernelBasis{Tk, Td} <: AbstractBasis where {Tk <: AbstractKernel, Td <: Real}
    nVariates::Int64
    nodes::Vector{Vector{Td}}
    kernel::Tk
end

"""
    KernelBasis(typeTd::Type{<:Real}, kernel::AbstractKernel, nVariates::Integer)
"""
function KernelBasis(typeTd::Type{<:Real}, kernel::AbstractKernel, nVariates::Integer)
    nodes = Vector{Vector{typeTd}}(undef, 0)
    KernelBasis(nVariates, nodes, kernel)
end

nVariates(p::KernelBasis) = p.nVariates
length(p::KernelBasis) = length(p.nodes)
getType(p::KernelBasis) = p.type
isDifferentiable(p::KernelBasis) = true
isTwiceDifferentiable(p::KernelBasis) = true


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


"""
    GaussianKernel

Define a Gaussian kernel
"""
struct GaussianKernel{Td} <: AbstractKernel  where {Td<:Real}
    sigmaSq::Td
    normalisation::Td
end

GaussianKernel(sigma::Td) where {Td<:Real} = GaussianKernel(sigma, sqrt(2 * pi * sigma^2))

"""
    kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}) where {Td<:Real}

Compute the value at `x` of the Gaussian kernel centered at `node`.
"""
function kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}) where {Td<:Real}
    norm2 = 0.
    for i in 1:length(node)
        norm2 += (node[i] - x[i])^2
    end
    return exp(-0.5  / k.sigmaSq * norm2) / k.normalisation^(length(node))
end

"""
    dkernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, derivativeIndex::Integer) where {Td<:Real}

Compute the first derivative w.r.t the `derivativeIndex` coordinate at `x` of the Gaussian kernel centered at `node`.
"""
function dkernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, derivativeIndex::Integer) where {Td<:Real}
    return - (x[derivativeIndex] - node[derivativeIndex]) / k.sigmaSq * kernel(k, node, x)
end

"""
    d2kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, derivativeIndex1::Integer, derivativeIndex2::Integer) where {Td<:Real}

Compute the second derivative w.r.t the (`derivativeIndex1`,`derivativeIndex2`)  coordinates at `x` of the Gaussian kernel centered at `node`.
"""
function d2kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, derivativeIndex1::Integer, derivativeIndex2::Integer) where {Td<:Real}
    d = (x[derivativeIndex1] - node[derivativeIndex1]) * (x[derivativeIndex2] - node[derivativeIndex2]) / k.sigmaSq
    if derivativeIndex2 == derivativeIndex1
        d -= 1.
    end
    return d / k.sigmaSq * kernel(k, node, x)
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
    secondDerivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, i1::Integer, i2::Integer)

Compute the value of the second derivative of the `index`-th basis function w.r.t to the `i1, i2` variates at point `x`.
"""
function secondDerivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, derivativeIndex1::Integer, derivativeIndex2::Integer)
    return d2kernel(b.kernel, b.nodes[index], x, derivativeIndex1, derivativeIndex2)
end
