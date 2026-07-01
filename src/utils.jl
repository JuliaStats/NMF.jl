# Numerical utilities to support implementation

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

normalize1!(a) = rmul!(a, 1 / sum(a))

function normalize1_cols!(a)
    for j = 1:size(a,2)
        normalize1!(view(a, :, j))
    end
end

function projectnn!(A::AbstractArray{T}) where T
    # project back all entries to non-negative domain
    @inbounds for i = 1:length(A)
        if A[i] < zero(T)
            A[i] = zero(T)
        end
    end
end

function posneg!(A::AbstractArray{T},
                 Ap::AbstractArray{T}, An::AbstractArray{T}) where T
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

function pdsolve!(A, x, uplo::Symbol=:U)
    # A must be positive definite
    # x <- inv(A) * x
    # A is overwritten by its Cholesky factor, x by the solution

    ldiv!(cholesky!(Hermitian(A, uplo)), x)
end

function pdrsolve!(A, B, x, uplo::Symbol=:U)
    # B must be positive definite
    # x <- A * inv(B)
    # B is overwritten by its Cholesky factor, x by the result

    Binv = inv(cholesky!(Hermitian(B, uplo)))
    mul!(x, A, Binv)
end

