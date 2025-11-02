# Transformations

```@docs
AbstractTransformation
```

The following methods are available and must be defined for any concrete subtype of [`AbstractTransformation`](@ref)

```@docs
apply!
jacobian
```

## Void transformation

It corresponds to $\varphi(x) = x$ for $x \in \mathbb{R}^d$.

```@docs
VoidTransformation
```

## Linear transformation

It corresponds to $\varphi: \mathbb{R}^d \to \mathbb{R}^d$ such that $\varphi(x) = (x - \alpha) * \sigma$ for where $\alpha, \sigma \in \mathbb{R}^d$ and `*` denotes a term by term multiplication. In the following, $\alpha$ is called the _center_ and $\sigma$ is called the _scale_

```@docs
LinearTransformation
getCenter
getScale
LinearTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
```

## Gaussian transformation

It corresponds to $\varphi: \mathbb{R}^d \to \mathbb{R}^d$ such that $\varphi(x) = \mathcal{N}((x - \alpha) / \sigma)$ where $\alpha, \sigma \in \mathbb{R}^d$, $\mathcal{N}$ is the cumulative distribution function of the standard normal distribution and `/` denotes a term by term multiplication.

```@docs
GaussianTransformation
getMean(::GaussianTransformation{<:Real})
getSigma(::GaussianTransformation{<:Real})
GaussianTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
```

## Log-normal transformation

It corresponds to $\varphi: \mathbb{R}^d \to \mathbb{R}^d$ such that $\varphi(x) = \mathcal{N}((\log(x) - \alpha) / \sigma)$ where $\alpha, \sigma \in \mathbb{R}^d$, $\mathcal{N}$ is the cumulative distribution function of the standard normal distribution and `/` denotes a term by term multiplication.

```@docs
LogNormalTransformation
getMean(::LogNormalTransformation{<:Real})
getSigma(::LogNormalTransformation{<:Real})
LogNormalTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
```
