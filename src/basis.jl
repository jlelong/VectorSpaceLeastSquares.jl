using SparseArrays

abstract type AbstractBasis end
function nVariates(::AbstractBasis) end
function size(::AbstractBasis) end

struct LinearTransform
    scale::Vector{<:Real}
    center::Vector{<:Real}
end

@enum PolynomialType begin
    Canonic
    Hermite
    Tchebitchev
end

"""
Represent a multivariate polynomial
- `degree`: total maximum degree
- `nVariates`: number of variates
- `dim`: dimension of the generated vector space
- `type`: a value from `PolynomialType`
- `tensor`: the sparse tensor representation of the polynomial. Polynomials are stored by column
"""
struct Polynomial <: AbstractBasis
    degree::Integer
    nVariates::Integer
    size::Integer
    type::PolynomialType
    tensor::SparseMatrixCSC{<:Integer, <:Integer} # Store polynomials by column
end

"""
Compute the full tensor representation of a multivariate polynomial
"""
function computePolynomialTensor(degree::Integer, nVariates::Integer)
    dim = binomial(nVariates + degree, degree)
    fullTensor = zeros(Int64, nVariates, dim)
    partialDegrees = zeros(Int64, nVariates)
    for i in 1:dim-1
        nextElementFound = false
        for j in nVariates:-1:1
            partialDegrees[j] += 1
            if sum(partialDegrees) <= degree
                nextElementFound = true
                break
            end
            partialDegrees[j] = 0
        end
        if !nextElementFound
            # We could not find the next element
            break
        end
        fullTensor[:, i] .= partialDegrees
    end
    return fullTensor
end


function Polynomial(degree::Integer, nVariates::Integer, type::PolynomialType)
    fullTensor = computePolynomialTensor(degree, nVariates)
    dim = Base.size(fullTensor, 1)
    return Polynomial(degree, nVariates, dim, type, sparse(fullTensor))
end

nVariates(p::Polynomial) = p.nVariates
size(p::Polynomial) = p.size

canonic1d(x::Real, n::Integer) = x^n
dcanonic1d(x::Real, n::Integer) = n == 0 ? 0. : x^(n-1)

"""
Recursive function to compute Hermite polynomials of any order.

- `x` evaluation point
- `n`` the order of the polynomial to be evaluated
- `n0` the rank of the initialization
- `f_n0` used to store the polynomial of order `n0`
- `f_n1` used to store the polynomial of order `n0 - 1`
"""
function hermite1d(x::Real, n::Integer, n0::Integer, f_n0::Real, f_n1::Real)
    if n == n0
      return f_n0
    else
        save = f_n0
        f_n0 = x * f_n0 - n0 * f_n1
        f_n1 = save
        return hermite1d(x, n, n0 + 1, f_n0, f_n1)
    end
end

function hermite1d(x::Real, n::Integer)
    if n == 0
        return 1
    elseif n == 1
        return x
    elseif n == 2
        return x * x - 1.
    elseif n == 3
        return (x * x - 3.) * x
    elseif n == 4
        x2 = x * x
        return (x2 - 6.) * x2 + 3
    elseif n == 5
        x2 = x * x
        return ((x2 - 10) * x2 + 15.) * x
    elseif n == 6
      x2 = x * x
      return ((x2 - 15.) * x2 + 45.) * x2 - 15.
    elseif n == 7
        x2 = x * x
        return (((x2 - 21.) * x2 + 105.) * x2 - 105) * x
    else
        f_n = hermite1d(x, 7)
        f_n_1 = hermite1d(x, 6)
        return hermite1d(x, n, 7, f_n, f_n_1)
    end
end

dhermite1d(x::Real, n::Integer) = n == 0 ? 0. : n * dhermite1d(x, n - 1)

"""
Recursive function to compute Tchebychev polynomials of any order.

- `x` evaluation point
- `n`` the order of the polynomial to be evaluated
- `n0` the rank of the initialization
- `f_n0` used to store the polynomial of order `n0`
- `f_n1` used to store the polynomial of order `n0 - 1`
"""
function tchebychev1d(n::Integer, n0::Integer, f_n0::Real, f_n1::Real)
    if n == 7
        return f_n0
    else
        save = f_n0
        f_n0 = 2 * x * f_n0 - f_n1
        f_n1 = save
        return tchebychev1d(x, n, n0 + 1, f_n0, f_n1)
    end
end

"""
Tchebychev polynomials of any order

- `x` the address of a real number
- `n` the order of the polynomial to be evaluated
"""
function tchebychev1d(x::Real, n::Integer)
    if n == 0
        return 1.
    elseif n == 1
        return x
    elseif n == 2
        return 2. * x * x - 1.
    elseif n == 2
        return (4. * x * x - 3.) * x
    elseif n == 2
        val2 = x * x
        return 8. * val2 * val2 - 8. * val2 + 1.
    elseif n == 2
        val2 = x * x
        val3 = val2 * x
        return 16. * val3 * val2 - 20. * val3 + 5. * x
    elseif n == 2
        val2 = x * x
        val4 = val2 * val2
        return 32. * val4 * val2 - 48. * val4 + 18. * val2 - 1
    elseif n == 2
        val2 = x * x
        val3 = val2 * x
        val4 = val2 * val2
        return (64. * val4 - 112. * val2 + 56) * val3 - 7. * x
    else
        f_n = tchebychevd1(x, 7)
        f_n_1 = tchebychevd1(x, 6)
        return tchebychev(x, n, 7, f_n, f_n_1)
    end
end



function value(polType::PolynomialType, degree::Integer, x::Real)
    if polType == Canonic
        return canonic1d(x, degree)
    elseif polType == Hermite
        return hermite1d(x, degree)
    else
        error("Not implemented")
    end
end

function differentiate(polType::PolynomialType, degree::Integer, x::Real)
    if polType == Canonic
        return dcanonic1d(x, degree)
    elseif polType == Hermite
        return dhermite1d(x, degree)
    else
        error("Not implemented")
    end
end

function value(polType::PolynomialType, partialDegrees::AbstractSparseVector{<:Integer, <:Integer}, x::AbstractVector{<:Real})
    val = 1.
    for r in nzrange(partialDegrees, 1)
        n = rowvals(partialDegrees)[r]
        d = nonzeros(partialDegrees)[r]
        val *= value(polType, d, x[n])
    end
    return val
end

value(p::Polynomial, x::AbstractVector{<:Real}, i::Integer) = value(p.type, p.tensor[:, i], x)

#
# Piecewise constant local basis
#

struct PiecewiseConstant <: AbstractBasis
    nVariates::Integer
    nIntervals::Vector{<:Integer}
    size::Integer
end

function PiecewiseConstant(nVariates::Integer, nIntervals::Integer)
    nIntervalsVect = Vector{Integer}(undef, nVariates)
    nIntervalsVect .= nIntervals
    dim = prod(nIntervals)
    return PiecewiseConstant(nVariates, nIntervalsVect, dim)
end

nVariates(p::PiecewiseConstant) = p.nVariates
size(p::PiecewiseConstant) = p.size
