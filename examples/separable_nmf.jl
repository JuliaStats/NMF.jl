using NMF

"""
separable_data(m,n,k)
Generate a (m x n) matrix X of nonnegative, separable data
with nonnegative rank k. The rows of X correspond to
observations and the columns of X correspond to features.
The separability condition implies that the columns of H
can be permuted to form a (k x k) diagonal block. Thus, the
(scaled) columns of W appear in A.
"""
function separable_data(m,n,k)
	# nonnegative factorization
	W = rand(m,k)
	H = rand(k,n)

	# impose separability
	for i = 1:k
		H[i,i] = 0.0
	end

	# permute columns of H
	H = H[:,randperm(n)]

	return W,H
end


m,n,k = 100,100,5
W,H = separable_data(m,n,k)

r = nnmf(W*H, k; alg=:spa)
