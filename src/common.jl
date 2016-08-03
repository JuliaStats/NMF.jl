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

immutable Result{T}
    W::Matrix{T}
    H::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T

    function Result(W::Matrix{T}, H::Matrix{T}, niters::Int, converged::Bool, objv)
        if size(W, 2) != size(H, 1)
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        end
        new(W, H, niters, converged, objv)
    end
end

# common algorithmic skeleton for iterative updating methods

abstract NMFUpdater{T}

function nmf_skeleton!{T}(updater::NMFUpdater{T},
                          X, W::Matrix{T}, H::Matrix{T},
                          maxiter::Int, verbose::Bool, tol)
    objv = convert(T, NaN)

    # init
    state = prepare_state(updater, X, W, H)
    preW = @compat Array{T,2}(size(W)...)
    preH = @compat Array{T,2}(size(H)...)
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
        copy!(preW, W)
        copy!(preH, H)

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
