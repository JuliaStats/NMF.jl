# common facilities

# tools to check size

function nmf_checksize(X, W::AbstractMatrix, H::AbstractMatrix)

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)

    if !(size(W,1) == p && size(H) == (k, n))
        throw(DimensionMismatch("Dimensions of X, W, and H are inconsistent."))
    end

    return (p, n, k)
end


# the result type

struct Result{T}
    W::Matrix{T}
    H::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T

    function Result{T}(W::Matrix{T}, H::Matrix{T}, niters::Int, converged::Bool, objv) where T
        if size(W, 2) != size(H, 1)
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        end
        new{T}(W, H, niters, converged, objv)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end

function nmf_skeleton!(updater::NMFUpdater{T},
                       X, W::Matrix{T}, H::Matrix{T},
                       maxiter::Int, verbose::Bool, tol) where T
    objv = convert(T, NaN)

    # init
    state = prepare_state(updater, X, W, H)
    preW = Matrix{T}(undef, size(W))
    preH = Matrix{T}(undef, size(H))
    if verbose
        objv = evaluate_objv(updater, state, X, W, H)
        @printf("%-5s     %-13s    %-13s    %-13s\n", "Iter", "objv", "objv.change", "(W & H).change")
        @printf("%5d    %13.6e\n", 0, objv)
    end

    # main loop
    converged = false
    t = 0
    while !converged && t < maxiter
        t += 1
        copyto!(preW, W)
        copyto!(preH, H)

        # update H
        update_wh!(updater, state, X, W, H)

        # determine convergence
        dev = max(maxad(preW, W), maxad(preH, H))
        if dev < tol
            converged = true
        end

        # display info
        if verbose
            preobjv = objv
            objv = evaluate_objv(updater, state, X, W, H)
            @printf("%5d    %13.6e    %13.6e    %13.6e\n",
                t, objv, objv - preobjv, dev)
        end
    end

    if !verbose
        objv = evaluate_objv(updater, state, X, W, H)
    end
    return Result{T}(W, H, t, converged, objv)
end
