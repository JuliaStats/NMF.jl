# Successive Projection Algorithm (SPA) for separable NMF
#
#   Reference: N. Gillis and S. A. Vavasis, "Fast and robust recursive
#   algorithms for separable nonnegative matrix factorization," 
#   IEEE Transactions on Pattern Analysis and Machine Intelligence, 
#   vol. 36, no. 4, pp. 698-714, 2013. 

struct SPA <: AbstractNMFAlgorithm
    obj::Symbol   # objective :mse or :div

    function SPA(;obj=:mse)
        obj == :mse || obj == :div || throw(ArgumentError("Invalid value for obj."))
        new(obj)
    end
end

# initialization
function spa(X::AbstractMatrix{T}, k::Integer; nnls_alg::Tuple{Symbol, Symbol}=(:pivot, :cache)) where T

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
    H = nonneg_lsq(W, X; alg=nnls_alg[1], variant=nnls_alg[2])
    projectnn!(H) 

    return W, H
end

# calculate statistics for result
function solve!(alg::SPA, X, W, H)
    T = eltype(W)
    if alg.obj == :mse
        objv = convert(T, 0.5) * sqL2dist(X, W*H)
    elseif alg.obj == :div
        objv = gkldiv(X, W*H)
    else
        error("Invalid value for obj.")
    end
    return Result{T}(W, H, 0, true, objv)
end

