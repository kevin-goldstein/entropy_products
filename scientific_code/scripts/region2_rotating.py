r"""
region2_rotating.py -- region-II stationarity for a rotating hairy black hole (Kerr-Sen with modulus).
For any xi = d_t + Omega d_phi, any smooth stationary-axisymmetric variation dPhi and an on-shell background,
    \int_{Sigma_II} xi . delta L = - \oint_{dSigma_II} xi . Theta(dPhi) = 0 ,
because xi vanishes on B+ (Omega = Omega+) and is tangent to B- (or vice versa).  For constant shifts of the
dilaton and of the axion this gives the 'averaged attractor equations' of the rotating black hole:
    \int_{r-}^{r+} dr \int dth sqrt(-g) [ 2 e^{-2chi} F^2 - 2 e^{4chi} (dxi)^2 ] = 0      (dilaton shift)
    \int_{r-}^{r+} dr \int dth sqrt(-g) F.(*F) = 0                                       (axion shift)
We evaluate both, and for contrast the same integrals over (r+, 3 r+), which do not vanish.
"""
import itertools
import sympy as sp
import mpmath as mp
from kerr_sen_check import g, Avec, chi, xi, params, r, th

mp.mp.dps = 20
X = [sp.Symbol('t'), r, th, sp.Symbol('phi')]
ginv = g.inv()
sg = sp.sqrt(-g.det())
F = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        F[m, n] = sp.diff(Avec[n], X[m]) - sp.diff(Avec[m], X[n])
Fup = ginv * F * ginv
FF = sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4))
# F.(*F) = F_mn (1/2) eps^{mnab} F_ab, eps^{0123} = -1/sg
FsF = 0
for p in itertools.permutations(range(4)):
    s = sp.combinatorics.Permutation(list(p)).signature()
    m, n, a, b = p
    FsF += s * (-1 / sg) * F[m, n] * F[a, b] / 2
dxi2 = sum(ginv[i, j] * sp.diff(xi, X[i]) * sp.diff(xi, X[j]) for i in (1, 2) for j in (1, 2))
import numpy as np
import checklib as ck
I_dil = sp.lambdify((r, th) + params, sg * (2 * sp.exp(-2 * chi) * FF - 2 * sp.exp(4 * chi) * dxi2), 'numpy')
I_ax = sp.lambdify((r, th) + params, sg * FsF, 'numpy')
xg, wg = np.polynomial.legendre.leggauss(160)


def dbl(fun, r0, r1, pv):
    rr = (r1 - r0) / 2 * xg + (r1 + r0) / 2
    tt = np.pi / 2 * xg + np.pi / 2
    R, T = np.meshgrid(rr, tt, indexing='ij')
    vals = fun(R, T, *pv)
    return (r1 - r0) / 2 * np.pi / 2 * np.einsum('i,j,ij->', wg, wg, vals)

pv = (1.4, 0.5, 0.8, 0.3)
Mn, an, Qn, cn = pv
r2n = np.exp(2 * cn) * Qn ** 2 / Mn
sq = np.sqrt((Mn - r2n / 2) ** 2 - an ** 2)
rp, rm = Mn - r2n / 2 + sq, Mn - r2n / 2 - sq
for name, fun in (('dilaton shift', I_dil), ('axion shift', I_ax)):
    inner = dbl(fun, rm, rp, pv)
    outer = dbl(fun, rp, 3 * rp, pv)
    print('%-14s: region II integral = % .3e ;  (r+, 3r+) integral = % .8f' % (name, inner, outer))
    ck.small('Kerr-Sen %s: region-II integral vanishes' % name, inner, 1e-10)
    if name == 'dilaton shift':
        ck.large('Kerr-Sen %s: the integral over (r+, 3r+) does not' % name, outer, 1e-2)
    else:
        ck.small('Kerr-Sen %s: the integral over (r+, 3r+) vanishes too (theta -> pi - theta parity)' % name, outer, 1e-12)
# the axion-shift integrand F.(*F) is odd under theta -> pi - theta, so its integral vanishes over any radial range;
# the non-trivial axion statements are the theta-dependent tests (e), (d1) of rotating_attractor_limit.py
ck.finish()
