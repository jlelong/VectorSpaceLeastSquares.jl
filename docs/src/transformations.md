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

It corresponds to $\varphi(x) = x$.

```@docs
VoidTransformation
```

## Linear transformation

It corresponds to $\varphi(x) = (x - \alpha) * \sigma$ where in the following $\alpha$ is called the _center_ and $\sigma$ is called the _scale_

```@docs
LinearTransformation
getCenter
getScale
LinearTransformation(x::AbstractVector{<:AbstractVector{T}}) where T<:Real
```

## Implementing a new transformation


