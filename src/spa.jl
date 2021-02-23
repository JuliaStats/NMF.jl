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
    # H = nonneg_lsq(W, X, alg=nnls_alg[1], variant=nnls_alg[2], gram=false)
    H = pivot_cache(W, X, gram=false) # temporarily used
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


#############################################################################################################
# The following code is included in NonNegLeastSquares.jl, and thus should be removed 
# when the registration is completed.

function pivot_cache(AtA,
    Atb::AbstractVector{T};
    tol::Float64=1e-8,
    max_iter=30*size(AtA,2)) where {T}


    # dimensions, initialize solution
    q = size(AtA,1)

    x = zeros(T, q) # primal variables
    y = -Atb    # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = BitArray(false for _ in 1:q)

    y[(!).(P)] = AtA[(!).(P),P]*x[P] - Atb[(!).(P)]

    # identify indices of infeasible variables
    V = @__dot__ (P & (x < -tol)) | (!P & (y < -tol))
    nV = sum(V)

    # while infeasible (number of infeasible variables > 0)
    while nV > 0

        if nV < β
            # infeasible variables decreased
            β = nV  # store number of infeasible variables
            α = 3   # reset α
        else
            # infeasible variables stayed the same or increased
            if α >= 1
                α = α-1 # tolerate increases for α cycles
            else
                # backup rule
                i = findlast(V)
                V = zeros(Bool,q)
                V[i] = true
            end
        end

        # update passive set
        #     P & ~V removes infeasible variables from P
        #     V & ~P  moves infeasible variables in ~P to P
        @__dot__ P = (P & !V) | (V & !P)

        # update primal/dual variables
        if !all(!, P)
            x[P] = _get_primal_dual(AtA, Atb, P)
        end
        #x[(!).(P)] = 0.0
        y[(!).(P)] = AtA[(!).(P),P]*x[P] - Atb[(!).(P)]
        #y[P] = 0.0

        # check infeasibility
        @__dot__ V = (P & (x < -tol)) | (!P & (y < -tol))
        nV = sum(V)
    end

    x[(!).(P)] .= zero(eltype(x))
    return x
end

@inline function _get_primal_dual(AtA, Atb, P)
    return pinv(AtA[P,P])*Atb[P]
end

## if multiple right hand sides are provided, solve each problem separately.
function pivot_cache(A,
     B::AbstractMatrix{T};
     gram::Bool = false,
     use_parallel::Bool = true,
     kwargs...) where {T}

    n = size(A,2)
    k = size(B,2)

    if gram
    # A,B are actually Gram matrices
    AtA = A
    AtB = B
    else
    # cache matrix computations
    AtA = A'*A
    AtB = A'*B
    end

    # compute result for each column
    if use_parallel && nprocs()>1
        X = @distributed (hcat) for i = 1:k
            pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
    else
        X = Array{T}(undef,n,k)
        for i = 1:k
            X[:,i] = pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
    end

    return X
end

#############################################################################################################
