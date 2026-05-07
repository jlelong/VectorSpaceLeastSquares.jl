# Examples

## Polynomial regression with non prior transformation

```julia
# Sample the data (x, y)
dim = 4
deg = 3
nSamples = 50000
T = Float32
f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
data = [randn(T, dim) for i in 1:nSamples]
y = f.(data)

# Build the VSLeastSquares object and solve the least squares problem
vslsq = VSLeastSquares(PolynomialBasis(Hermite, dim, deg), VoidTransformation(), T)
fit(vslsq, data, y)

# Compute new predictions
x = randn(T, dim)
predict(vslsq, x)
```

## Polynomial regression with a prior linear transformation

```julia
# Sample the data (x, y)
dim = 4
deg = 3
nSamples = 10000
f(x) = 2 * x[2]^3 - x[1] * x[4] + 7 * x[3]^2 * x[1]
data = [1.0 .+ 2.0 .* randn(T, dim) for i in 1:nSamples]
y = f.(data)

# Create an automatic linear transformation, which uses the empirical mean and standard deviation
transformation = LinearTransformation(data)

# Build the VSLeastSquares object and solve the least squares problem
vslsq = VSLeastSquares(PolynomialBasis(Hermite, dim, deg), transformation, T)
fit(vslsq, data, y)

# Compute new predictions and their gradient
x = randn(T, dim)
predict(vslsq, x)
gradient(vslsq, x)
```

## Piecewise constant regression with non prior transformation

```julia
nIntervals = 50
nSamples = 100000
dim = 2
data = [rand(Float64, dim) for i in 1:nSamples]
f(x) = log(1. + sum(x.^2))
y = f.(data)
vslsq = VSLeastSquares(PiecewiseConstantBasis(dim, nIntervals), VoidTransformation(), Float64)
fit(vslsq, data, y)
x = rand(T, dim)
predict(vslsq, x)
```

## Kernel regression example

```julia
noise = 0.2
X = range(-5, 5; length=100)
Y = sin.(X) + noise * randn((100))
gaussianKernel = GaussianKernel(0.5)
kernelBasis = KernelBasis(Float64, gaussianKernel, 1)
vslsq = VSLeastSquares(kernelBasis, VoidTransformation(), Float64)
XTest = range(-6, 6; length=500)
fit(vslsq, [[x] for x in X], Y, 0.25)
predictions = predict.(vslsq, [[x] for x in XTest])
trueValues = sin.(XTest)
plot(XTest, [predictions, trueValues], ylims=[-2, 2], label=["prediction" "true value"] )
```
