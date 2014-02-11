# Numerical utilities to support implementation

import Base.LAPACK: potrf!, potri!, potrs!

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

function projectnn!(A::AbstractArray{Float64})
	# project back all entries to non-negative domain
	@inbounds for i = 1:length(A)
		if A[i] < 0.0
			A[i] = 0.0
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

