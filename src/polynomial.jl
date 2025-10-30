# Polynomial basis
using SparseArrays: nonzeros, rowvals, nzrange

"""
    PolynomialType

List the families of polynomials available through the [`PolynomialBasis`](@ref) type
- `Canonic`
- `Hermite`
- `Tchebychev`
"""
@enum PolynomialType begin
    Canonic
    Hermite
    Tchebychev
end

"""
    PolynomialBasis

Represent a multivariate polynomial
- `degree::Int64`: maximum total degree
- `nVariates::Int6`: number of variates
- `dim::Int6`: dimension of the generated vector space
- `type::PolynomialType`: a value from `PolynomialType`
- `tensor::SparseMatrixCSC{Int64, Int64}`: the sparse tensor representation of the polynomial. Polynomials are stored by column.
"""
struct PolynomialBasis <: AbstractBasis
    degree::Int64
    nVariates::Int64
    size::Int64
    type::PolynomialType
    tensor::SparseMatrixCSC{Int64, Int64} # Store polynomials by column
end

nVariates(p::PolynomialBasis) = p.nVariates
length(p::PolynomialBasis) = p.size
getType(p::PolynomialBasis) = p.type
getTensor(p::PolynomialBasis) = p.tensor

"""
    computePolynomialTensor(nVariates::Integer, degree::Integer)

Compute the full tensor representation of a multivariate polynomial
"""
function computePolynomialTensor(nVariates::Integer, degree::Integer)
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


"""
    PolynomialBasis(type::PolynomialType, nVariates::Integer, degree::Integer)

Create a polynomial basis with `type`, `nVariates` variables and total maximum `degree`
"""
function PolynomialBasis(type::PolynomialType, nVariates::Integer, degree::Integer)
    fullTensor = computePolynomialTensor(nVariates, degree)
    dim = size(fullTensor, 2)
    return PolynomialBasis(degree, nVariates, dim, type, sparse(fullTensor))
end

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
        x2 = x * x
        return 8. * x2 * x2 - 8. * x2 + 1.
    elseif n == 5
        x2 = x * x
        x3 = x2 * x
        return 16. * x3 * x2 - 20. * x3 + 5. * x
    elseif n == 6
        x2 = x * x
        x4 = x2 * x2
        return 32. * x4 * x2 - 48. * x4 + 18. * x2 - 1
    elseif n == 7
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        return (64. * x4 - 112. * x2 + 56) * x3 - 7. * x
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
    value(polType::PolynomialType, degree::Integer, x::Real)

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
        error("Unknown polynomial type: $polType")
    end
end

"""
    derivative(polType::PolynomialType, degree::Integer, x::Real)

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
        error("Unknown polynomial type: $polType")
    end
end

"""
    value(p::PolynomialBasis, x::AbstractVector{<:Real}, i::Integer)

Evaluate the `i`-th function of a Polynomial basis `p` at point `x`
"""
function value(p::PolynomialBasis, x::AbstractVector{<:Real}, i::Integer)
    val = 1.
    T = getTensor(p)
    for r in nzrange(T, i)
        n = rowvals(T)[r]
        deg = nonzeros(T)[r]
        val = val * value(getType(p), deg, x[n])
    end
    return val
end


"""
    derivative(p::PolynomialBasis, x::AbstractVector{Td}, polIndex::Ti, derivativeIndex::Ti) where {Td<:Real, Ti<:Integer}

Evaluate the first partial derivative w.r.t variable `derivativeIndex` of the `polIndex`-th member of the polynomial basis `p`
"""
function derivative(p::PolynomialBasis, x::AbstractVector{Td}, polIndex::Ti, derivativeIndex::Ti) where {Td<:Real, Ti<:Integer}
    T = getTensor(p)
    if T[derivativeIndex, polIndex] == 0
        return 0.
    end
    val = 1.
    for r in nzrange(T, polIndex)
        n = rowvals(T)[r]
        deg = nonzeros(T)[r]
        if n == derivativeIndex
            val *= derivative(getType(p), deg, x[n])
        else
            val *= value(getType(p), deg, x[n])
        end
    end
    return val
end

