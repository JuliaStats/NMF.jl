# NMF for dense matrices

using NMF
using Printf
using Random

function run(algname)

    # prepare data
    Random.seed!(5678)
    p = 8
    k = 5
    n = 100

    Wg = abs.(randn(p, k))
    Hg = abs.(randn(k, n))
    X = Wg * Hg + 0.1 * randn(p, n)

    # run NNMF
    println("Algorithm: $(algname)")
    println("---------------------------------")

    r = nnmf(X, k; 
             init=:nndsvdar,
             alg=Symbol(algname), 
             maxiter=30, 
             verbose=true)

    # display results
    println("numiters  = $(r.niters)")
    println("converged = $(r.converged)")
    @printf("objvalue  = %.6e\n", r.objvalue)
    println("W matrix = ")
    NMF.printf_mat(r.W)

    println()
end


function print_help()
    println("Usage:")
    println()
    println("  julia densenmf.jl <alg>")
    println()
    println("  <alg> is the name of the chosen algorithm, which can be ")
    println()
    println("    multmse:   Multiplicative update (minimize MSE)")
    println("    multdiv:   Multiplicative update (minimize divergence)")
    println("    projals:   Projected ALS")
    println("    alspgrad:  ALS Projected Gradient Descent")
    println("    cd:        Coordinate Descent")
    println("    greedycd:  Greedy Coordinate Descent")
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
        @warn("Invalid command line arguments.")
        print_help()
    end
end

main(ARGS)

