# Vector Space bases

The vector space $\mathcal{V}$ is represented by the abstract type `AbstractBasis`.

```@docs
AbstractBasis
nVariates(B::AbstractBasis)
length(B::AbstractBasis)
size(B::AbstractBasis)
getType(B::AbstractBasis)
value(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer)
derivative(B::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)
```

## Polynomial Bases

Polynomial Bases are implemented using the `PolynomialBasis` type

```@docs
PolynomialBasis
```

The following families of polynomials are available

```@docs
PolynomialType
```

A [`PolynomialBasis`](@ref) object can be created using

```@docs
PolynomialBasis(type::PolynomialType, nVariates::Integer, degree::Integer)
```
