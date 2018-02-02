using NonNegLeastSquares

# Empty type, 
type SPA{T} 
	obj::Symbol   # objective :mse or :div

	function SPA{T}( ; obj=:mse) where T
            obj == :mse || obj == :div || error("Invalid value for obj.")
            new(obj)
        end
end

# initialization
function spa(
	X::Matrix,
	k::Int;
	nnls_alg = (:pivot,:cache)
	) 

    # Normalize data so that columns of X sum to one
    Rscale=sum(X,1)
    R = X ./ Rscale
    
    # W = R[:,ai], where ai are the "anchor indices"
    # (ai forms the convex hull of columns in R)
    ai = (Int64)[]

    # Add columns of X that are furthest from span(W)
    for j = 1:k
	# Add column with the largest residual
	push!(ai, indmax(sum(R.^2,1)))
        
	# Project R onto the selected column
	p = R[:,ai[j]]         # column we're projecting on
	R -= p*(p'*R) ./(p'*p)  # new residual matrix
    end

    # Estimate W as the anchor columns of X
    W = X[:,ai]

    # Estimate H by non-negative least squares: minimize ||X - W*H||
    H = nonneg_lsq(W,X,alg=nnls_alg[1],variant=nnls_alg[2],gram=false)

    return W,H
end

# calculate statistics for result
function solve!(alg::SPA, X, W, H)
	if alg.obj == :mse
		objv = sqL2dist(X, W*H)
	elseif alg.obj == :div
		objv = gkldiv(X, W*H)
	else
		error("Invalid value for obj.")
	end
        ret = Result(W, H, 0, true, objv)
	return ret
end
