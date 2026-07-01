## NMF.jl

A Julia package for non-negative matrix factorization (NMF).

[![CI](https://github.com/JuliaStats/NMF.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/JuliaStats/NMF.jl/actions/workflows/ci.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaStats.github.io/NMF.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaStats.github.io/NMF.jl/dev)
[![CodeCov](https://codecov.io/github/JuliaStats/NMF.jl/badge.svg?branch=master)](https://codecov.io/github/JuliaStats/NMF.jl?branch=master)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

---------------------------

## Overview

*Non-negative matrix factorization (NMF)* factorizes a non-negative matrix `X`
into the product of two lower-rank non-negative matrices `W` and `H`, such that
`W*H` optimally approximates `X` in some sense. Such techniques are widely used
in text mining, image analysis, and recommendation systems.

A factorization proceeds in two stages: an *initialization* function produces
starting factors `W` and `H`, and an *optimization* algorithm refines them. The
high-level [`nnmf`](https://JuliaStats.github.io/NMF.jl/stable) function runs
both stages in a single call.

Apart from `nnmf`, the package's names are not exported; refer to them with the
`NMF.` prefix (for example `NMF.solve!`, `NMF.randinit`).

## Installation

```julia
using Pkg
Pkg.add("NMF")
```

## Quick start

```julia
using NMF

X = rand(8, 6)                     # a non-negative matrix
r = nnmf(X, 3; alg=:multmse, maxiter=50, tol=1.0e-4)
W, H = r.W, r.H                    # the factors of the approximation X ≈ W*H
```

See the [documentation](https://JuliaStats.github.io/NMF.jl/stable) for the full
list of initialization methods and factorization algorithms, their options, and
further examples.

## Development status

**Note:** Non-negative matrix factorization is an area of active research. New
algorithms are proposed every year. Contributions are very welcome.

Implemented:

- Lee & Seung's multiplicative update (for both MSE and divergence objectives)
- (Naive) projected alternating least squares
- ALS projected gradient methods
- Coordinate descent methods
- Random initialization
- NNDSVD initialization
- Sparse NMF
- Separable NMF

Planned:

- Probabilistic NMF

See also [NMFMerge](https://github.com/HolyLab/NMFMerge.jl), which can augment
any least-squares NMF algorithm.
