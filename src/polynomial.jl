# Polynomial basis

@enum PolynomialType begin
    Canonic
    Hermite
    Tchebychev
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
- `f_n0_1` used to store the polynomial of order `n0 - 1`
"""
function hermite1d(x::Real, n::Integer, n0::Integer, f_n0::Real, f_n0_1::Real)
    if n == n0
      return f_n0
    else
        save = f_n0
        f_n0 = x * f_n0 - n0 * f_n0_1
        f_n0_1 = save
        return hermite1d(x, n, n0 + 1, f_n0, f_n0_1)
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
        n0 = 7
        f_n0 = hermite1d(x, n0)
        f_n0_1 = hermite1d(x, n0 - 1)
        return hermite1d(x, n, 7, f_n0, f_n0_1)
    end
end

dhermite1d(x::Real, n::Integer) = n == 0 ? 0. : n * hermite1d(x, n - 1)

"""
Recursive function to compute Tchebychev polynomials of any order.

- `x` evaluation point
- `n`` the order of the polynomial to be evaluated
- `n0` the rank of the initialization
- `f_n0` used to store the polynomial of order `n0`
- `f_n0_1` used to store the polynomial of order `n0 - 1`
"""
function tchebychev1d(x::Real, n::Integer, n0::Integer, f_n0::Real, f_n0_1::Real)
    if n == n0
        return f_n0
    else
        save = f_n0
        f_n0 = 2 * x * f_n0 - f_n0_1
        f_n0_1 = save
        return tchebychev1d(x, n, n0 + 1, f_n0, f_n0_1)
    end
end

"""
Tchebychev polynomials of any order

- `x` the evaluation point
- `n` the order of the polynomial to be evaluated
"""
function tchebychev1d(x::Real, n::Integer)
    if n == 0
        return 1.
    elseif n == 1
        return x
    elseif n == 2
        return 2. * x * x - 1.
    elseif n == 3
        return (4. * x * x - 3.) * x
    elseif n == 4
        val2 = x * x
        return 8. * val2 * val2 - 8. * val2 + 1.
    elseif n == 5
        val2 = x * x
        val3 = val2 * x
        return 16. * val3 * val2 - 20. * val3 + 5. * x
    elseif n == 6
        val2 = x * x
        val4 = val2 * val2
        return 32. * val4 * val2 - 48. * val4 + 18. * val2 - 1
    elseif n == 7
        val2 = x * x
        val3 = val2 * x
        val4 = val2 * val2
        return (64. * val4 - 112. * val2 + 56) * val3 - 7. * x
    else
        f_n = tchebychev1d(x, 7)
        f_n_1 = tchebychev1d(x, 6)
        return tchebychev1d(x, n, 7, f_n, f_n_1)
    end
end


"""
Recursive computation of the first derivative of the Tchebychev polynomials of any order.

- `x` the evaluation point
- `n` the order of the polynomial to be evaluated
- `n0` the rank of initialization
- `f_n` the derivative of the polynomial of order `n0`.
- `f_n_1` the derivative of the polynomial of order `n0 - 1`
"""
function dtchebychev1d(x::Real, n::Integer, n0::Integer, f_n0::Real, f_n0_1::Real)
    if n == n0
        return f_n0
    else
        save = f_n0
        f_n0 = 2 * x * Real(n0 + 1) / Real(n0) * (f_n0) - Real(n0 + 1) / Real(n0 - 1) * (f_n0_1)
        f_n0_1 = save
        return dtchebychev1d(x, n, n0 + 1, f_n0, f_n0_1)
    end
end

"""
First derivative of the Tchebytchev polynomials

- `x` the evaluation point
- `n` the index of the polynomial whose first derivative is to be evaluated
"""
function dtchebychev1d(x::Real, n::Integer)
    if n == 0
        return 0.
    elseif n == 1
        return 1.
    elseif n == 2
        return 4. * x
    elseif n == 3
        return (12. * x * x - 3.)
    elseif n == 4
        return (32. * x * x - 16.) * x
    elseif n == 5
        x2 = x * x
        return 80. * x2 * x2 - 60. * x2 + 5.
    elseif n == 6
        x2 = x * x
        x4 = x2 * x2
        return (192. * x4 - 192. * x2 + 36.) * x
    elseif n == 7
        x2 = x * x
        x4 = x2 * x2
        return (448. * x4 - 560. * x2 + 168) * x2 - 7.
    else
        n0 = 7
        f_n0 = dtchebychev1d(x, n0)
        f_n0_1 = dtchebychev1d(x, n0 - 1)
        return dtchebychev1d(x, n, n0, f_n0, f_n0_1)
    end
end

"""
Evaluate a 1d polynomial of type `polType` and degree `degree` at `x`
"""
function value(polType::PolynomialType, degree::Integer, x::Real)
    if polType == Canonic
        return canonic1d(x, degree)
    elseif polType == Hermite
        return hermite1d(x, degree)
    elseif polType == Tchebychev
        return tchebychev1d(x, degree)
    else
        error("Not implemented")
    end
end

"""
Evaluate the first derivative of a 1d polynomial of type `polType` and degree `degree` at `x`
"""
function derivative(polType::PolynomialType, degree::Integer, x::Real)
    if polType == Canonic
        return dcanonic1d(x, degree)
    elseif polType == Hermite
        return dhermite1d(x, degree)
    elseif polType == Tchebychev
        return dtchebychev1d(x, degree)
    else
        error("Not implemented")
    end
end

"""
Evaluate a multivariate polynomial of type `polType` at `x`.

- `partialDegrees` a sparse vector describing the partial degrees
"""
function value(polType::PolynomialType, partialDegrees::AbstractSparseVector{<:Integer, <:Integer}, x::AbstractVector{<:Real})
    val = 1.
    for r in nzrange(partialDegrees, 1)
        n = rowvals(partialDegrees)[r]
        deg = nonzeros(partialDegrees)[r]
        val *= value(polType, deg, x[n])
    end
    return val
end

"""
Evaluate the `i`-th function of a Polynomial basis `p` at point `x`
"""
value(p::Polynomial, x::AbstractVector{<:Real}, i::Integer) = value(p.type, p.tensor[:, i], x)

"""
Evaluate the first derivative of a multivariate polynomial of type `polType` at `x`.

- `partialDegrees` is a sparse vector describing the partial degrees
- `derivativeIndex` is the index of the partial derivative
"""
function derivative(polType::PolynomialType, partialDegrees::AbstractSparseVector{<:Integer, <:Integer}, derivativeIndex::Integer, x::AbstractVector{<:Real})
    if partialDegrees[derivativeIndex] == 0
        return 0.
    end
    val = 1.
    for r in nzrange(partialDegrees, 1)
        n = rowvals(partialDegrees)[r]
        deg = nonzeros(partialDegrees)[r]
        if n == derivativeIndex
            val *= derivative(polType, deg, x[n])
        else
            val *= value(polType, deg, x[n])
        end
    end
    return val
end

"""
Evaluate the first partial derivative w.r.t variable `derivativeIndex` of the `polIndex`-th member of the polynomial basis `p`
"""
derivative(p::Polynomial, x::AbstractVector{<:Real}, polIndex::Integer, derivativeIndex::Integer) = derivative(p.type, p.tensor[:, polIndex], derivativeIndex, x)

