# Numerical utilities to support implementation

import Base.BLAS: nrm2
import Base.LAPACK: potrf!, potri!, potrs!

function printf_mat(x::AbstractMatrix)
    @inbounds for i = 1:size(x,1)
        for j = 1:size(x,2)
            @printf("%8.4f ", x[i,j])
        end
        println()
    end
end

function mul!(y::AbstractArray, x::AbstractArray, c::Number)
    n = length(x)
    length(y) == n || error("Inconsistent lengths.")
    for i = 1:n
        @inbounds y[i] = c * x[i]
    end
    y
end

function adddiag!(A::Matrix, a::Number)
    m, n = size(A)
    m == n || error("A must be square.")
    if a != 0.0
        for i = 1:m
            @inbounds A[i,i] += a
        end
    end
    return A
end

normalize1!(a) = scale!(a, 1 / sum(a))

function normalize1_cols!(a)
    for j = 1:size(a,2)
        normalize1!(view(a, :, j))
    end
end

function projectnn!{T}(A::AbstractArray{T})
    # project back all entries to non-negative domain
    @inbounds for i = 1:length(A)
        if A[i] < zero(T)
            A[i] = zero(T)
        end
    end
end

function posneg!{T}(A::AbstractArray{T},
                    Ap::AbstractArray{T}, An::AbstractArray{T})
    # decompose A into positive part Ap and negative part An
    # s.t. A = Ap - An

    n = length(A)
    length(Ap) == length(An) == n || error("Input dimensions mismatch.")

    @inbounds for i = 1:n
        ai = A[i]
        if ai >= zero(T)
            Ap[i] = ai
            An[i] = zero(T)
        else
            Ap[i] = zero(T)
            An[i] = -ai
        end
    end
end

function pdsolve!(A, x, uplo::Char='U')
    # A must be positive definite
    # x <- inv(A) * x
    # both A and x will be overriden

    potrf!(uplo, A)
    potrs!(uplo, A, x)
end

function pdrsolve!(A, B, x, uplo::Char='U')
    # B must be positive definite
    # x <- A * inv(B)
    # both B and x will be overriden

    # inverse B in place
    potrf!(uplo, B)
    potri!(uplo, B)
    copytri!(B, uplo)

    # x <- A * B (the inversed one)
    A_mul_B!(x, A, B)
end

