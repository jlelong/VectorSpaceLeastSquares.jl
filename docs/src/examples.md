# Examples

## Polynomial regression with non prior transformation

```julia
dim = 4
deg = 3
nSamples = 50000
T = Float32
f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
data = [randn(T, dim) for i in 1:nSamples]
y = f.(data)
vslsq = VSLeastSquares(PolynomialBasis(deg, dim, Hermite), VoidTransformation(), T)
fit(vslsq, data, y)
x = randn(T, dim)
predict(vslsq, x)
```
