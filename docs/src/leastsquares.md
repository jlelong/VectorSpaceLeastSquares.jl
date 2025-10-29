# Least squares problem

Solving a least squares problem of the form

$$\inf_{\alpha \in \mathbb{R}^d} \sum_{m=1}^M \left(\sum_{i=1}^d \alpha_i g_i\circ\varphi(x_m) - y_m\right)^2$$

is done by creating an instance of `VSLeastSquares`

```@docs
VSLeastSquares
VSLeastSquares(basis::Tb, transform::Tt=VoidTransformation(), Td::Type=Float64) where {Tb<:AbstractBasis, Tt<:AbstractTransformation}
length(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
nVariates(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
size(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
getCoefficients(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
getBasis(vslsq::VSLeastSquares{Tb, Tt, Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
```

An instance of `VSLeastSquares` is typically manipulated using the following methods to solve the least squares problem and compute a prediction

```@docs
fit(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{<:AbstractVector{Td}}, y::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
predict(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
```

If the underlying basis of type `Tb` is differentiable the derivative of the prediction can be computed using

```@docs
derivative(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}, index::Integer) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
gradient(vslsq::VSLeastSquares{Tb, Tt, Td}, x::AbstractVector{Td}) where {Tb<:AbstractBasis, Tt<:AbstractTransformation, Td<:Real}
```
