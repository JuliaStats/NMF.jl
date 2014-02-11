## NMF.jl

A Julia package for non-negative matrix factorization (NMF).

---------------------------

## Development Status

#### Done

- Lee & Seung's Multiplicative Update (for both MSE & Divergence objectives)
- Alternate Least Square (ALS) 
- Random Initialization

#### To do

- NNDSVD Initilization
- Projected Gradient Methods
- NMF with sparsity regularization
- Probabilistic NMF

## Overview

*Non-negative Matrix Factorization (NMF)* generally refers to the techniques for factorizing a non-negative matrix ``X`` into the product of two lower rank matrices ``W`` and ``H``, such that ``WH`` optimally approximates ``X`` in some sense. Such techniques are widely used in text mining, image analysis, and recommendation systems. 

This package provides two sets of tools, respectively for *initilization* and *optimization*. A typical NMF procedure consists of two steps: (1) use an initilization function that initialize ``W`` and ``H``; and (2) use an optimization algorithm to pursue the optimal solution.

Most types and functions in this package are not exported. Users are encouraged to use them with the prefix ``NMF.``. This way allows us to use shorter names within the package and makes the codes more explicit and clear on the user side.

## Initialization

- **NMF.randinit**(X, k[; zeroh=false])

    Initialize ``W`` and ``H`` given the input matrix ``X`` and the rank ``k``. This function returns a pair ``(W, H)``. 

    Suppose the size of ``X`` is ``(p, n)``, then the size of ``W`` and ``H`` are respectively ``(p, k)`` and ``(k, n)``.

    Usage:

    ```julia
    W, H = NMF.randinit(X, 3)
    ```

    For some algorithms (*e.g.* ALS), only ``W`` needs to be initialized. For such cases, one may set the keyword argument ``zeroh``to be ``true``, then in the output ``H`` will be simply a zero matrix of size ``(k, n)``.


## Factorization Algorithms

This package provides multiple factorization algorithms. Each algorithm corresponds to a type. One can create an algorithm *instance* by choosing a type and specifying the options, and run the algorithm using ``NMF.solve!``:

#### The NMF.solve! Function

**NMF.solve!**(alg, X, W, H)

Use the algorithm ``alg`` to factorize ``X`` into ``W`` and ``H``. 

Here, ``W`` and ``H`` must be pre-allocated matrices (respectively of size ``(p, k)`` and ``(k, n)``). ``W`` and ``H`` must be appropriately initialized before this function is invoked. For some algorithms, both ``W`` and ``H`` must be initialized (*e.g.* multiplicative updating); while for others, only ``W`` needs to be initialized (*e.g.* ALS).

The matrices ``W`` and ``H`` are updated in place.

In general, this function returns a result instance of type ``NMF.Result``, which is defined as

```julia
immutable Result
    W::Matrix{Float64}    # W matrix
    H::Matrix{Float64}    # H matrix
    niters::Int           # number of elapsed iterations
    converged::Bool       # whether the optimization procedure converges
    objvalue::Float64     # objective value of the last step
end
```

#### Algorithms

- **Multiplicative Updating**

    Reference: Daniel D. Lee and H. Sebastian Seung. Algorithms for Non-negative Matrix Factorization. Advances in NIPS, 2001.

    This algorithm has two different kind of objectives: minimizing mean-squared-error (``:mse``) and minimizing divergence (``:div``). Both ``W`` and ``H`` need to be initialized.

    ```julia
    MultUpdate(obj=:mse,        # objective, either :mse or :div
               maxiter=100,     # maximum number of iterations
               verbose=false,   # whether to show procedural information
               tol=1.0e-6,      # tolerance of changes on W and H upon convergence
               lambda=1.0e-9)   # regularization coefficients (added to the denominator)
    ```

    **Note:** the values above are default values for the keyword arguments. One can override part (or all) of them.


- **Naive Alternate Least Square**

    This algorithm alternately updates ``W`` and ``H`` while holding the other fixed. Each update step solves ``W`` or ``H`` without enforcing the non-negativity constrait, and forces all negative entries to zeros afterwards. Only ``W`` needs to be initialized. 

    ```julia
    NaiveALS(maxiter::Integer=100,    # maximum number of iterations
             verbose::Bool=false,     # whether to show procedural information
             tol::Real=1.0e-6,        # tolerance of changes on W and H upon convergence
             lambda_w::Real=1.0e-6,   # L2 regularization coefficient for W
             lambda_h::Real=1.0e-6)   # L2 regularization coefficient for H
    ```

## Examples

Here are examples that demonstrate how to use this package to factorize a non-negative dense matrix.

#### Use Multiplicative Update

```julia
import NMF

... # prepare X

 # initialize
W, H = NMF.randinit(X, 5)

 # optimize 
NMF.solve!(NMF.MultUpdate(obj=:mse,maxiter=100), X, W, H)
```

#### Use Naive ALS

```julia
import NMF

... # prepare X

 # initialize
W, H = NMF.randinit(X, 5)

 # optimize 
NMF.solve!(NMF.NaiveALS(maxiter=50), X, W, H)
```

