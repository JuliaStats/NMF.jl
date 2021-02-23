# Successive Projection Algorithm (SPA) for separable NMF
#
#   Reference: N. Gillis and S. A. Vavasis, "Fast and robust recursive
#   algorithms for separable nonnegative matrix factorization," 
#   IEEE Transactions on Pattern Analysis and Machine Intelligence, 
#   vol. 36, no. 4, pp. 698-714, 2013. 

mutable struct SPA{T}
    obj::Symbol   # objective :mse or :div

    function SPA{T}(;obj=:mse) where T
        obj == :mse || obj == :div || throw(ArgumentError("Invalid value for obj."))
        new{T}(obj)
    end
end

"""
separable_data(m,n,k)
Generate a (m x n) matrix X of nonnegative, separable data
with nonnegative rank k. The rows of X correspond to
observations and the columns of X correspond to features.
The separability condition implies that the columns of H
can be permuted to form a (k x k) diagonal block, and 
the sum of the entries of each column of H is at most one. 
Thus, the (scaled) columns of W appear in X.
"""
function separable_data(m, n, k)
    W = rand(m, k)
    
    # impose separability
    V = rand(k, n-k)
    V ./= sum(V, dims=1)
    H = [Matrix(I, k, k) V]
    # permute columns of H
    H = H[:, randperm(n)]
    
    return W, H
end

# initialization
function spa(X::Matrix{T}, k::Integer; nnls_alg::Tuple{Symbol, Symbol}=(:pivot, :cache)) where T

    # Normalize data so that columns of X sum to one
    R = X ./ sum(X, dims=1)

    # W = R[:,ai], where ai are the "anchor indices"
    # (ai forms the convex hull of columns in R)
    ai = Vector{Int}(undef, k)

    # Add columns of X that are furthest from span(W)
    for j = 1:k
        # Add column with the largest residual
        ai[j] = argmax(vec(sum(R.^2, dims=1)))
        
        # Project R onto the selected column
        p = R[:,ai[j]]         	# column we're projecting on
        R -= p*(p'*R) ./(p'*p) 	# new residual matrix
    end
    
    # Estimate W as the anchor columns of X
    W = X[:,ai]
    
    # Estimate H by non-negative least squares: minimize ||X - W*H||
    H = nonneg_lsq(W, X, alg=nnls_alg[1], variant=nnls_alg[2], gram=false)
    projectnn!(H) 

    return W, H
end

# calculate statistics for result
function solve!(alg::SPA{T}, X, W, H) where T
    if alg.obj == :mse
        objv = convert(T, 0.5) * sqL2dist(X, W*H)
    elseif alg.obj == :div
        objv = gkldiv(X, W*H)
    else
        error("Invalid value for obj.")
    end
    return Result{T}(W, H, 0, true, objv)
end
