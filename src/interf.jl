# Interface function: nnmf

function nnmf{T}(X::AbstractMatrix{T}, k::Integer;
                 init::Symbol=:nndsvdar,
                 alg::Symbol=:alspgrad,
                 maxiter::Integer=100,
                 tol::Real=cbrt(eps(T)/100),
                 verbose::Bool=false)
    
    p, n = size(X)
    k <= min(p, n) || error("The value of k should not exceed min(size(X)).")

    # determine whether H needs to be initialized
    if alg == :projals 
        initH = false
    else
        initH = true
    end

    # perform initialization
    if alg == :spa
        W, H = spa(X, k)
    elseif init == :spa
        W, H = spa(X, k) 
    elseif init == :random
        W, H = randinit(X, k; zeroh=!initH, normalize=true)
    elseif init == :nndsvd
        W, H = nndsvd(X, k; zeroh=!initH)
    elseif init == :nndsvda
        W, H = nndsvd(X, k; variant=:a, zeroh=!initH)
    elseif init == :nndsvdar
        W, H = nndsvd(X, k; variant=:ar, zeroh=!initH)
    else
        error("Invalid value for init.")
    end

    # choose algorithm
    alginst = 
        alg == :projals ? ProjectedALS{T}(maxiter=maxiter, tol=tol, verbose=verbose) :
        alg == :alspgrad ? ALSPGrad{T}(maxiter=maxiter, tol=tol, verbose=verbose) :
        alg == :multmse ? MultUpdate{T}(obj=:mse, maxiter=maxiter, tol=tol, verbose=verbose) :
        alg == :multdiv ? MultUpdate{T}(obj=:div, maxiter=maxiter, tol=tol, verbose=verbose) :
        alg == :spa ? SPA{T}(obj=:mse) :
        error("Invalid algorithm.")

    # run optimization
    solve!(alginst, X, W, H)
end

