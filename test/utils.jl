# Test internal utilities functions

import NMF
using Base.Test

make_pdmat(n::Int) = (g = randn(n, n); NMF.adddiag!(g'g, 0.1))

## adddiag!

a0 = rand(3, 3)
a = copy(a0)

NMF.adddiag!(a, 0.)
@test a == a0

NMF.adddiag!(a, 2.5)
@test a == a0 + 2.5 * eye(3,3)

## projectnn!

a0 = randn(5, 5)
a = copy(a0)
NMF.projectnn!(a)
@test a == max(a0, 0.0)

## posneg!

a = randn(5, 5)
ac = copy(a)
ap = zeros(size(a))
an = zeros(size(a))

NMF.posneg!(a, ap, an)
@test a == ac
@test ap == max(a, 0.0)
@test an == max(-a, 0.0)

## pdsolve!

A = make_pdmat(5)
X = rand(5, 3)
Y = A * X
NMF.pdsolve!(A, Y)
@test_approx_eq X Y

## pdrsolve!

B = make_pdmat(5)
X = rand(4, 5)
Y = X * B
Xr = zeros(4, 5)
NMF.pdrsolve!(Y, B, Xr)
@test_approx_eq Xr X
