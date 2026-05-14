# Kernel Bases

It is possible to use a kernel basis instead of a basis of functions of a single variate (be it multi-dimensional): the functions are defined by kernels $K(x_i, \cdot) : \mathbb{R}^d \to \mathbb{R}$, where the nodes $x_i \in \mathbb{R}^d$ are defined as the training values

```@docs
KernelBasis
KernelBasis(typeTd::Type{<:Real}, kernel::AbstractKernel, nVariates::Integer)
```

## Kernel type

To create a `KernelBasis`, you must define the kernel to use, which is done by deriving a concrete type from `AbstractKernel`

```@docs
AbstractKernel
kernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real})
dkernel(k::AbstractKernel, node::AbstractVector{<:Real}, x::AbstractVector{<:Real}, derivativeIndex::Integer)
```

### Gaussian kernel

The Gaussian kernel is defined by
$$K(x,y) = \frac{1}{(2 \pi \sigma^2)^{d/2}} e^{-\frac{|x - y|^2}{2 \sigma^2}}.$$

```@docs
GaussianKernel
kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}) where {Td<:Real}
dkernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, i::Integer) where {Td<:Real}
d2kernel(k::GaussianKernel{Td}, node::AbstractVector{Td}, x::AbstractVector{Td}, i::Integer, j::Integer) where {Td<:Real}
```
