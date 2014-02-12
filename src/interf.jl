# Interface function: nnmf

function nnmf(X::Matrix{Float64}, k::Integer; 
			  init::Symbol=:nndsvdar, 
			  alg::Symbol=:projals, 
			  maxiter::Integer=100,
			  tol::Real=1.0e-6, 
			  verbose::Bool=false)

	p, n = size(X)
	k <= min(p, n) || error("The value of k should not exceed min(size(X)).")

	# determine whether H needs to be initialized
	if alg == :projals 
		initH = false
	elseif alg == :multmse || alg == :multdiv
		initH = true
	else
		error("Invalid value for alg.")
	end

	# perform initialization
	if init == :random
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
		alg == :projals ? ProjectedALS(maxiter=maxiter, tol=tol, verbose=verbose) :
		alg == :multmse ? MultUpdate(obj=:mse, maxiter=maxiter, tol=tol, verbose=verbose) :
		alg == :multdiv ? MultUpdate(obj=:div, maxiter=maxiter, tol=tol, verbose=verbose) :
		error("Invalid algorithm.")

	# run optimization
	solve!(alginst, X, W, H)
end

