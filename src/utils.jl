# Numerical utilities to support implementation

import Base.BLAS: nrm2
import Base.LAPACK: potrf!, potri!, potrs!

function printf_mat(x::ContiguousMatrix{Float64})
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            @printf("%8.4f ", x[i,j])
        end
        println()
    end
end

function mul!(y::ContiguousArray{Float64}, x::ContiguousArray{Float64}, c::Float64)
    n = length(x)
    length(y) == n || error("Inconsistent lengths.")
    for i = 1:n
        @inbounds y[i] = c * x[i]
    end
    y
end

function adddiag!(A::Matrix{Float64}, a::Float64)
    m, n = size(A)
    m == n || error("A must be square.")
    if a != 0.0
        for i = 1:m
            @inbounds A[i,i] += a
        end
    end
    return A
end

normalize1!(a::ContiguousVector{Float64}) = scale!(a, 1.0 / sum(a))

function normalize1_cols!(a::DenseArray{Float64,2})
    for j = 1:size(a,2)
        normalize1!(view(a, :, j))
    end
end

function projectnn!(A::AbstractArray{Float64})
    # project back all entries to non-negative domain
    @inbounds for i = 1:length(A)
        if A[i] < 0.0
            A[i] = 0.0
        end
    end
end

function posneg!(A::ContiguousArray{Float64}, 
                 Ap::ContiguousArray{Float64}, An::ContiguousArray{Float64})
    # decompose A into positive part Ap and negative part An
    # s.t. A = Ap - An

    n = length(A)
    length(Ap) == length(An) == n || error("Input dimensions mismatch.")

    @inbounds for i = 1:n
        ai = A[i]
        if ai >= 0.0
            Ap[i] = ai
            An[i] = 0.0
        else
            Ap[i] = 0.0
            An[i] = -ai
        end
    end
end

function pdsolve!(A::Matrix{Float64}, x::VecOrMat{Float64}, uplo::Char='U')
    # A must be positive definite
    # x <- inv(A) * x
    # both A and x will be overriden

    potrf!(uplo, A)
    potrs!(uplo, A, x)
end

function pdrsolve!(A::Matrix{Float64}, B::Matrix{Float64}, x::Matrix{Float64}, uplo::Char='U')
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

