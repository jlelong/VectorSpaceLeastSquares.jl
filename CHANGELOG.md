# Release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.2] 2026-05-17

### Fixed

- Fix standard variance mismatch when creating a Gaussian kernel.

## [1.4.1] 2026-05-16

### Changed

- Improve computational efficiency for kernel regression.

## [1.4.0] 2026-05-14

### Added

- Compute the second derivative of a predict. Only available for void or linear transformations.

## [1.3.0] 2026-05-07

### Added

- Kernelized least squares.

## [1.2.0] 2026-04-21

### Added

- Ridge regularization

## [1.1.0] 2025-11-02

### Added

- Gaussian and Log-Normal transformations are added.
- A basis `PiecewiseConstantBasis` to represent piecewise constant functions on a predefined grid: the generating functions are the characteristic functions of each cell.
- An online documentation is available at [https://jlelong.github.io/VectorSpaceLeastSquares.jl](https://jlelong.github.io/VectorSpaceLeastSquares.jl)
- Add a method `isDifferentiable(::AbstractBasis)`
