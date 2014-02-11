# NMF for dense matrices

using NMF

function run(algname)

    # choose algorithm
    alg = algname == "mult-mse" ? NMF.MultUpdate(obj=:mse, maxiter=30, verbose=true) :
          algname == "mult-div" ? NMF.MultUpdate(obj=:div, maxiter=30, verbose=true) :
          algname == "projals" ? NMF.ProjectedALS(maxiter=30, verbose=true) :
          error("Invalid algorithm name.")

    # prepare data
    p = 8
    k = 5
    n = 100

    Wg = abs(randn(p, k))
    Hg = abs(randn(k, n))
    X = Wg * Hg + 0.1 * randn(p, n)

    # initialize
    W0, H0 = NMF.nndsvd(X, k; variant=:ar)

    println("Algorithm: $(algname)")
    println("---------------------------------")

    # run optimization
    r = NMF.solve!(alg, X, W0, H0)

    println("numiters  = $(r.niters)")
    println("converged = $(r.converged)")
    @printf("objvalue  = %.6e\n", r.objvalue)
    println("W matrix = ")
    W = r.W
    for i = 1:p
        for j = 1:k
            @printf("%8.4f ", W[i,j])
        end
        println()
    end

    println()
end


function print_help()
    println("Usage:")
    println()
    println("  julia densenmf.jl <alg>")
    println()
    println("  <alg> is the name of the chosen algorithm, which can be ")
    println()
    println("    mult-mse:  multiplicative update (minimize MSE)")
    println("    mult-div:  multiplicative update (minimize divergence)")
    println("    projals:   Projected ALS")
    println()
end

function main(args)
    if isempty(args)
        print_help()
    elseif length(args) == 1
        a = lowercase(args[1])
        if a == "-h" || a == "--help"
            print_help()
        else
            run(a)
        end
    else
        warn("Invalid command line arguments.")
        print_help()
    end
end

main(ARGS)

