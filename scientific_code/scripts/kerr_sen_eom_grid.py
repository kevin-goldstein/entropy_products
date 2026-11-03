r"""
kerr_sen_eom_grid.py -- field-equation residuals of Kerr-Sen with the modulus switched on, on a grid of points
outside the outer horizon, in region II and inside the inner horizon.
"""
import mpmath as mp
import sympy as sp
from kerr_sen_check import g, Avec, scal, gsig, fcoup, ftcoup, params
from eom_num import make_residuals
import checklib as ck

mp.mp.dps = 20
res = make_residuals(g, Avec, scal, gsig, fcoup, ftcoup, lambda s: sp.Integer(0), params)
for pv in [(mp.mpf('1.4'), mp.mpf('0.5'), mp.mpf('0.8'), mp.mpf('0.3')), (mp.mpf('2.0'), mp.mpf('0.9'), mp.mpf('1.1'), mp.mpf('-0.4'))]:
    Mn, an, Qn, cn = pv
    r2 = mp.e ** (2 * cn) * Qn ** 2 / Mn
    sq = mp.sqrt((Mn - r2 / 2) ** 2 - an ** 2)
    rp, rm = Mn - r2 / 2 + sq, Mn - r2 / 2 - sq
    worst = 0
    for rr in (rm / 2, (rp + rm) / 2, 1.5 * rp, 5 * rp):
        for tt in (mp.mpf('0.3'), mp.mpf('1.1'), mp.mpf('2.0')):
            E, Mx, S = res(rr, tt, pv)
            worst = ck.max_abs((worst, E, Mx, S))
    print('params (M,a,Q,chi0) =', [mp.nstr(x, 3) for x in pv], ' r- =', mp.nstr(rm, 5), ' r+ =', mp.nstr(rp, 5),
          ' max residual over 12 points (incl. region II and r < r-):', mp.nstr(worst, 3))
    ck.small('Kerr-Sen field equations at 12 points, (M,a,Q,chi0) = %s' % [mp.nstr(x, 3) for x in pv], worst, 1e-15)
# negative control: the gauge field rescaled by 1.1 at fixed metric and scalars
bad = make_residuals(g, [1.1 * x for x in Avec], scal, gsig, fcoup, ftcoup, lambda s: sp.Integer(0), params)
E, Mx, S = bad(mp.mpf('2.9'), mp.mpf('1.1'), (mp.mpf('1.4'), mp.mpf('0.5'), mp.mpf('0.8'), mp.mpf('0.3')))
ck.large('negative control: A -> 1.1 A violates the field equations', ck.max_abs((E, Mx, S)), 1e-4)
ck.finish()
