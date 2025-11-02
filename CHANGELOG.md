# Release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Gaussian and Log-Normal transformations are added.
- A basis `PiecewiseConstantBasis` to represent piecewise constant functions on a predefined grid: the generating functions are the characteristic functions of each cell.
- An online documentation is available at [https://jlelong.github.io/VectorSpaceLeastSquares.jl](https://jlelong.github.io/VectorSpaceLeastSquares.jl)
- Add a method `isDifferentiable(::AbstractBasis)`
