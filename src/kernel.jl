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
    type::Type{Tk}
end

"""
    KernelBasis(type::Type{Tk}, nodes::Vector{Vector{Td}}) where {Tk <: AbstractKernel, Td <: Real}

Build a kernel basis using `nodes`. The number of kernels in the basis is equal to the number of `nodes`.
"""
function KernelBasis(type::Type{Tk}, nodes::Vector{Vector{Td}}) where {Tk <: AbstractKernel, Td <: Real}
    KernelBasis(length(nodes[1]), length(nodes), nodes, type)
end

function KernelBasis(typeTk::Type{Tk}, typeTd::Type{Td}, nVariates::Integer) where {Tk <: AbstractKernel, Td <: Real}
    nodes = Vector{Vector{typeTd}}(undef, 0)
    KernelBasis(nVariates, nodes, typeTk)
end

nVariates(p::KernelBasis) = p.nVariates
length(p::KernelBasis) = length(p.nodes)
getType(p::KernelBasis) = p.type
isDifferentiable(p::KernelBasis) = true


"""
    kernel(::Type{<:AbstractKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Compute the value at `x` of the kernel centered at `node`.
"""
function kernel(::Type{<:AbstractKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}) end

"""
    dkernel(::Type{<:AbstractKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)

Compute the first derivative w.r.t the `derivativeIndex` coordinate at `x` of the kernel centered at `node`.
"""
function dkernel(::Type{<:AbstractKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer) end


struct GaussianKernel <: AbstractKernel end

"""
    kernel(::Type{GaussianKerne}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})

Compute the value at `x` of the Gaussian kernel centered at `node`.
"""
function kernel(::Type{GaussianKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return exp(0.5 * sum((node .- x).^2)) / sqrt(2 * pi)
end

"""
    dkernel(::Type{GaussianKerne}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)

Compute the first derivative w.r.t the `derivativeIndex` coordinate at `x` of the Gaussian kernel centered at `node`.
"""
function dkernel(::Type{GaussianKernel}, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)
    return (x[derivativeIndex] - node[derivativeIndex]) * kernel(GaussianKernel, node, x)
end

"""
    value(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer)

Compute the value of the kernel centered around the `index`-th node at point `x`.
"""
function value(b::KernelBasis{Tk, Td}, x::AbstractVector{Td}, index::Integer) where {Tk <: AbstractKernel, Td <: Real}
    return kernel(b.type, b.nodes[index], x)
end

"""
    derivative(b::KernelBasis{<:AbstractKernel, <:Real}, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)

Compute the value of the first derivative of the kernel centered around the `index`-th node w.r.t to the `derivativeIndex` coordinate at point `x`.
"""
function derivative(b::KernelBasis{Tk, Td}, x::AbstractVector{Td}, index::Integer, derivativeIndex::Integer)  where {Tk <: AbstractKernel, Td <: Real}
    return dkernel(b.type, b.nodes[index], x, derivativeIndex)
end
