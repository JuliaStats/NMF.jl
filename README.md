## NMF.jl

A Julia package for non-negative matrix factorization (NMF).

[![Build Status](https://travis-ci.org/JuliaStats/NMF.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/NMF.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaStats/NMF.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaStats/NMF.jl?branch=master)

---------------------------

## Development Status

**Note:** Nonnegative Matrix Factorization is an area of active research. New algorithms are proposed every year. Contributions are very welcomed.

#### Done

- Lee & Seung's Multiplicative Update (for both MSE & Divergence objectives)
- (Naive) Projected Alternate Least Squared
- ALS Projected Gradient Methods
- Random Initialization
- NNDSVD Initialization
- Sparse NMF

#### To do

- Probabilistic NMF


## Overview

*Non-negative Matrix Factorization (NMF)* generally refers to the techniques for factorizing a non-negative matrix ``X`` into the product of two lower rank matrices ``W`` and ``H``, such that ``WH`` optimally approximates ``X`` in some sense. Such techniques are widely used in text mining, image analysis, and recommendation systems. 

This package provides two sets of tools, respectively for *initilization* and *optimization*. A typical NMF procedure consists of two steps: (1) use an initilization function that initialize ``W`` and ``H``; and (2) use an optimization algorithm to pursue the optimal solution.

Most types and functions (except the high-level function ``nnmf``) in this package are not exported. Users are encouraged to use them with the prefix ``NMF.``. This way allows us to use shorter names within the package and makes the codes more explicit and clear on the user side.


## High-Level Interface

The package provides a high-level function ``nnmf`` that runs the entire procedure (initialization + optimization):

**nnmf**(X, k, ...)

This function factorizes the input matrix ``X`` into the product of two non-negative matrices ``W`` and ``H``. 

In general, it returns a result instance of type ``NMF.Result``, which is defined as

```julia
struct Result
    W::Matrix{Float64}    # W matrix
    H::Matrix{Float64}    # H matrix
    niters::Int           # number of elapsed iterations
    converged::Bool       # whether the optimization procedure converges
    objvalue::Float64     # objective value of the last step
end
```

The function supports the following keyword arguments:

- ``init``:  A symbol that indicates the initialization method (default = ``:nndsvdar``). 

    This argument accepts the following values:

    - ``random``:  matrices filled with uniformly random values
    - ``nndsvd``:  standard version of NNDSVD
    - ``nndsvda``:  NNDSVDa variant
    - ``nndsvdar``:  NNDSVDar variant  
                
- ``alg``:  A symbol that indicates the factorization algorithm (default = ``:alspgrad``).

    This argument accepts the following values:

    - ``multmse``:  Multiplicative update (using MSE as objective)
    - ``multdiv``:  Multiplicative update (using divergence as objective)
    - ``projals``:  (Naive) Projected Alternate Least Square
    - ``alspgrad``:  Alternate Least Square using Projected Gradient Descent
    - ``cd``: Coordinate Descent solver that uses Fast Hierarchical Alternating Least Squares (implemetation similar to scikit-learn)

- ``maxiter``: Maximum number of iterations (default = ``100``).

- ``tol``: tolerance of changes upon convergence (default = ``1.0e-6``).

- ``verbose``: whether to show procedural information (default = ``false``).



## Initialization

- **NMF.randinit**(X, k[; zeroh=false, normalize=false])

    Initialize ``W`` and ``H`` given the input matrix ``X`` and the rank ``k``. This function returns a pair ``(W, H)``. 

    Suppose the size of ``X`` is ``(p, n)``, then the size of ``W`` and ``H`` are respectively ``(p, k)`` and ``(k, n)``.

    Usage:

    ```julia
    W, H = NMF.randinit(X, 3)
    ```

    For some algorithms (*e.g.* ALS), only ``W`` needs to be initialized. For such cases, one may set the keyword argument ``zeroh``to be ``true``, then in the output ``H`` will be simply a zero matrix of size ``(k, n)``.

    Another keyword argument is ``normalize``. If ``normalize`` is set to ``true``, columns of ``W`` will be normalized such that each column sum to one.

- **NMF.nndsvd**(X, k[; zeroh=false, variant=:std])

    Use the *Non-Negative Double Singular Value Decomposition (NNDSVD)* algorithm to initialize ``W`` and ``H``. 

    Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.

    Usage:

    ```julia
    W, H = NMF.nndsvd(X, k)
    ```

    This function has two keyword arguments:

    - ``zeroh``: have ``H`` initialized when it is set to ``true``, or set ``H`` to all zeros when it is set to ``false``.
    - ``variant``: the variant of the algorithm. Default is ``std``, meaning to use the standard version, which would generate a rather sparse ``W``. Other values are ``a`` and ``ar``, respectively corresponding to the variants: *NNDSVDa* and *NNDSVDar*. Particularly, ``ar`` is recommended for dense NMF.


## Factorization Algorithms

This package provides multiple factorization algorithms. Each algorithm corresponds to a type. One can create an algorithm *instance* by choosing a type and specifying the options, and run the algorithm using ``NMF.solve!``:

#### The NMF.solve! Function

**NMF.solve!**(alg, X, W, H)

Use the algorithm ``alg`` to factorize ``X`` into ``W`` and ``H``. 

Here, ``W`` and ``H`` must be pre-allocated matrices (respectively of size ``(p, k)`` and ``(k, n)``). ``W`` and ``H`` must be appropriately initialized before this function is invoked. For some algorithms, both ``W`` and ``H`` must be initialized (*e.g.* multiplicative updating); while for others, only ``W`` needs to be initialized (*e.g.* ALS).

The matrices ``W`` and ``H`` are updated in place.


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


- **(Naive) Projected Alternate Least Square**

    This algorithm alternately updates ``W`` and ``H`` while holding the other fixed. Each update step solves ``W`` or ``H`` without enforcing the non-negativity constrait, and forces all negative entries to zeros afterwards. Only ``W`` needs to be initialized. 

    ```julia
    ProjectedALS(maxiter::Integer=100,    # maximum number of iterations
                 verbose::Bool=false,     # whether to show procedural information
                 tol::Real=1.0e-6,        # tolerance of changes on W and H upon convergence
                 lambda_w::Real=1.0e-6,   # L2 regularization coefficient for W
                 lambda_h::Real=1.0e-6)   # L2 regularization coefficient for H
    ```

- **Alternate Least Square Using Projected Gradient Descent**

    Reference: Chih-Jen Lin. Projected Gradient Methods for Non-negative Matrix Factorization. Neural Computing, 19 (2007).

    This algorithm adopts the alternate least square strategy. A efficient projected gradient descent method is used to solve each sub-problem. Both ``W`` and ``H`` need to be initialized.

    ```julia
    ALSPGrad(maxiter::Integer=100,      # maximum number of iterations (in main procedure)
             maxsubiter::Integer=200,   # maximum number of iterations in solving each sub-problem
             tol::Real=1.0e-6,          # tolerance of changes on W and H upon convergence
             tolg::Real=1.0e-4,         # tolerable gradient norm in sub-problem (first-order optimality)
             verbose::Bool=false)       # whether to show procedural information
    ```

- **Coordinate Descent solver with Fast Hierarchical Alternating Least Squares**

    Reference: Cichocki, Andrzej, and P. H. A. N. Anh-Huy. Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721 (2009).
    
    Sequential constrained minimization on a set of squared Euclidean distances over W and H matrices. Uses l_1 and l_2 penalties to enforce sparsity.

    ```julia
    CoordinateDescent(maxiter::Integer=100,      # maximum number of iterations (in main procedure)
                      verbose::Bool=false,       # whether to show procedural information
                      tol::Real=1.0e-6,          # tolerance of changes on W and H upon convergence
                      α::Real=0.0,               # constant that multiplies the regularization terms
                      regularization=:both,      # select whether the regularization affects the components (H), the transformation (W), both or none of them (:components, :transformation, :both, :none)
                      l₁ratio::Real=0.0,         # l1 / l2 regularization mixing parameter (in [0; 1])
                      shuffle::Bool=false)       # if true, randomize the order of coordinates in the CD solver
    ```

## Examples

Here are examples that demonstrate how to use this package to factorize a non-negative dense matrix.

#### Use High-level Function: nnmf

```julia
... # prepare input matrix X

r = nnmf(X, k; alg=:multmse, maxiter=30, tol=1.0e-4)

W = r.W
H = r.H
```

#### Use Multiplicative Update

```julia
import NMF

 # initialize
W, H = NMF.randinit(X, 5)

 # optimize 
NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=100), X, W, H)
```

#### Use Naive ALS

```julia
import NMF

 # initialize
W, H = NMF.randinit(X, 5)

 # optimize 
NMF.solve!(NMF.ProjectedALS{Float64}(maxiter=50), X, W, H)
```

#### Use ALS with Projected Gradient Descent

```julia
import NMF

 # initialize
W, H = NMF.nndsvdar(X, 5)

 # optimize 
NMF.solve!(NMF.ALSPGrad{Float64}(maxiter=50, tolg=1.0e-6), X, W, H)
```

#### Use Coordinate Descent

```julia
import NMF

 # initialize
W, H = NMF.nndsvdar(X, 5)

 # optimize 
NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50, α=0.5, l₁ratio=0.5), X, W, H)
```

