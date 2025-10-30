# Vector Space bases

The vector space $\mathcal{V}$ is represented by the abstract type `AbstractBasis`.

```@docs
AbstractBasis
nVariates(b::AbstractBasis)
length(b::AbstractBasis)
size(b::AbstractBasis)
getType(b::AbstractBasis)
value(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer)
derivative(b::AbstractBasis, x::AbstractVector{<:Real}, index::Integer, derivativeIndex::Integer)
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

## Piecewise constant functions (local bases)

Piecewise constant functions can be efficiently obtained from a basis of local functions with disjoint supports. Such bases are implemented using the `PiecewiseConstantBasis`

```@docs
PiecewiseConstantBasis
```

A [`PiecewiseConstantBasis`](@ref) object can be created using

```@docs
PiecewiseConstantBasis(nVariates::Integer, nIntervals::Integer)
PiecewiseConstantBasis(nVariates::Integer, nIntervals::Vector{<:Integer})
```
