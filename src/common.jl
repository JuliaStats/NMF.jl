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


Base.:(==)(A::Result, B::Result) = A.W == B.W && A.H == B.H && A.niters == B.niters && A.converged == B.converged && A.objvalue == B.objvalue
Base.hash(s::Result, h::UInt) = hash(s.objvalue, hash(s.converged, hash(s.niters, hash(s.H, hash(s.W, h + (0x09c9f08cfcba6de3 % UInt))))))


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
        start = time()
        objv = evaluate_objv(updater, state, X, W, H)
        @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(W & H).relchange")
        @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
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
        converged, dev = stop_condition(W, preW, H, preH, tol)

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, X, W, H)
            @printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
                t, elapsed, objv, objv - preobjv, dev)
        end
    end

    if !verbose
        objv = evaluate_objv(updater, state, X, W, H)
    end
    return Result{T}(W, H, t, converged, objv)
end


function stop_condition(W::AbstractArray{T}, preW::AbstractArray, H::AbstractArray, preH::AbstractArray, eps::AbstractFloat) where T
    devmax = zero(T)
    for j in axes(W,2)
        dev_w = sum_w = zero(T)
        for i in axes(W,1)
            dev_w += (W[i,j] - preW[i,j])^2
            sum_w += (W[i,j] + preW[i,j])^2
        end
        dev_h = sum_h = zero(T)
        for i in axes(H,2)
            dev_h += (H[j,i] - preH[j,i])^2
            sum_h += (H[j,i] + preH[j,i])^2
        end
        devmax = max(devmax, sqrt(max(dev_w/sum_w, dev_h/sum_h)))
        if sqrt(dev_w) > eps*sqrt(sum_w) || sqrt(dev_h) > eps*sqrt(sum_h)
            return false, devmax
        end
    end
    return true, devmax
end
