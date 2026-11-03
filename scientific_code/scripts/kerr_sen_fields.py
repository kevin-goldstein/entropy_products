r"""
kerr_sen_fields.py -- Kerr-Sen solution of the axion-dilaton theory, in the conventions of the paper:
   L = (1/16pi)[R - 2(dchi)^2 - (1/2) e^{4 chi}(dxi)^2 - e^{-2chi} F^2] eps + (1/16pi) 2 kappa xi F^F
i.e. scalars (chi, xi), g_ij = diag(1, e^{4chi}/4), f = e^{-2chi}, ftilde = kappa xi.
Normalisations (c1, c2, kappa) are fixed below by demanding that the field equations hold.
"""
import itertools
import sympy as sp
import mpmath as mp
from eom_num import make_residuals, r, th
import checklib as ck

M, a, q = sp.symbols('M a q', positive=True)
c1, c2, kap = sp.symbols('c1 c2 kappa')


def fields(c1v, c2v):
    r2 = c1v * q ** 2 / M
    St = r * (r + r2) + a ** 2 * sp.cos(th) ** 2
    Dl = r * (r + r2) - 2 * M * r + a ** 2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * M * r / St)
    g[0, 3] = g[3, 0] = -2 * a * M * r * sp.sin(th) ** 2 / St
    g[3, 3] = sp.sin(th) ** 2 * (r * (r + r2) + a ** 2 + 2 * M * r * a ** 2 * sp.sin(th) ** 2 / St)
    g[1, 1] = St / Dl
    g[2, 2] = St
    chi = sp.log((r ** 2 + a ** 2 * sp.cos(th) ** 2) / St) / 2
    xi = c2v * (q ** 2 / M) * a * sp.cos(th) / (r ** 2 + a ** 2 * sp.cos(th) ** 2)
    A = [-q * r / St, sp.Integer(0), sp.Integer(0), q * r * a * sp.sin(th) ** 2 / St]
    return g, A, [chi, xi]


gsig = lambda s: sp.diag(1, sp.exp(4 * s[0]) / 4)
fcoup = lambda s: sp.exp(-2 * s[0])
Vpot = lambda s: sp.Integer(0)

if __name__ == '__main__':
    mp.mp.dps = 20
    pv = (mp.mpf('1.4'), mp.mpf('0.5'), mp.mpf('0.8'))
    pt = (mp.mpf('3.3'), mp.mpf('0.9'))
    best = []
    for c1v in (sp.Rational(1, 2), 1, 2):
        for c2v in (sp.Rational(1, 2), 1, 2, -1, -sp.Rational(1, 2), -2):
            for kv in (1, -1, 2, -2, sp.Rational(1, 2), -sp.Rational(1, 2)):
                g, A, sc = fields(c1v, c2v)
                res = make_residuals(g, A, sc, gsig, fcoup, lambda s, kv=kv: kv * s[1], Vpot, (M, a, q))
                E, Mx, S = res(pt[0], pt[1], pv)
                tot = ck.max_abs(E) + ck.max_abs(Mx) + ck.max_abs(S)
                best.append((tot, c1v, c2v, kv))
    best.sort(key=lambda x: x[0])
    for b in best[:5]:
        print('residual %s  for c1=%s c2=%s kappa=%s' % (mp.nstr(b[0], 3), b[1], b[2], b[3]))
    ck.true('the best normalisation is c1 = 1, c2 kappa = -1 (c2 = 1, kappa = -1 or c2 = -1, kappa = 1)',
            {(b[1], b[2], b[3]) for b in best[:2]} == {(1, 1, -1), (1, -1, 1)})
    ck.small('field equations hold for c1 = 1, c2 = 1, kappa = -1', best[0][0], 1e-18)
    ck.large('every other normalisation in the scan fails the field equations', best[2][0], 1e-5)
    ck.finish()
