r"""
dilaton_exact.py -- exact non-extremal dyonic-type dilaton black hole (GJN "case 1"):
    L = (1/16pi)[R - 2 (dphi)^2 - e^{-2phi} (F^1)^2 - e^{2phi} (F^2)^2],  electric charges Q1, Q2
    V_eff = e^{2phi} Q1^2 + e^{-2phi} Q2^2
Solution (to be verified):
    e^{2phi} = e^{2phi0} (r+S)/(r-S),  B^2 = r^2 - S^2,  A = (r-rp)(r-rm)/(r^2-S^2),  rp,rm = M +- Delta
We verify the field equations, solve the two algebraic constraints, and then test every
statement of the covariant-phase-space analysis on this family.
"""
import sympy as sp
from check_K_conservation import ricci
from cps import inv_metric

r, t, th, vp = sp.symbols('r t theta varphi')
M, S, Dl, p0 = sp.symbols('M Sigma Delta phi0', real=True)
Q1, Q2 = sp.symbols('Q1 Q2', positive=True)

phi = p0 + sp.log((r + S) / (r - S)) / 2
B2 = r ** 2 - S ** 2
rp, rm = M + Dl, M - Dl
A = (r - rp) * (r - rm) / B2
f1 = sp.exp(-2 * phi)   # coupling of F^1
f2 = sp.exp(2 * phi)    # coupling of F^2
E1 = Q1 / (f1 * B2)     # A^1_t' (4D: E = f^{-1} Q / B^2)
E2 = Q2 / (f2 * B2)


def field_equations_residuals():
    x = [t, r, th, vp]
    g = sp.diag(-A, 1 / A, B2, B2 * sp.sin(th) ** 2)
    ginv = inv_metric(g)
    Ric = ricci(g, x)
    res = []
    for (m, n) in [(0, 0), (1, 1), (2, 2)]:
        T = 2 * sp.diff(phi, x[m]) * sp.diff(phi, x[n])
        for (fa, Ea) in [(f1, E1), (f2, E2)]:
            F = sp.zeros(4, 4); F[1, 0] = Ea; F[0, 1] = -Ea
            FmFn = sum(F[m, p] * F[n, q] * ginv[p, q] for p in range(4) for q in range(4))
            FF = sum(F[a, b] * F[c, d] * ginv[a, c] * ginv[b, d] for a in range(4) for b in range(4) for c in range(4) for d in range(4))
            T += 2 * fa * FmFn - g[m, n] * fa * FF / 2
        res.append(sp.simplify((Ric[m, n] - T) * ginv[m, m]))
    # scalar: 4 box phi - f1' F1^2 - f2' F2^2 = 0 ; F^2 = -2E^2 for electric fields
    box = sp.diff(B2 * A * sp.diff(phi, r), r) / B2
    scal = 4 * box - (-2 * f1) * (-2 * E1 ** 2) - (2 * f2) * (-2 * E2 ** 2)
    res.append(sp.simplify(scal))
    return res


if __name__ == '__main__':
    res = field_equations_residuals()
    # the residuals are rational in r; demand all coefficients vanish
    conds = []
    for e in res:
        num = sp.numer(sp.together(sp.expand(e)))
        num = sp.expand(num)
        conds += sp.Poly(num, r).coeffs()
    conds = sorted({sp.factor(c) for c in conds if c != 0}, key=sp.default_sort_key)
    print('number of independent conditions:', len(conds))
    for c in conds:
        print('   ', c)
    import checklib as ck
    fam_ = {S: (sp.exp(-2 * p0) * Q2 ** 2 - sp.exp(2 * p0) * Q1 ** 2) / (2 * M)}
    fam_[Dl] = sp.sqrt(M ** 2 + fam_[S] ** 2 - sp.exp(2 * p0) * Q1 ** 2 - sp.exp(-2 * p0) * Q2 ** 2)
    ck.true('all field equations vanish on the family (2 M Sigma = e^{-2phi0} Q2^2 - e^{2phi0} Q1^2, '
            'Delta^2 = M^2 + Sigma^2 - Veff)', all(sp.simplify(c.subs(fam_)) == 0 for c in conds))


# ---------------------------------------------------------------------------------------------
# covariant phase space checks on the exact family, independent parameters (M, phi0, Q1, Q2)
# ---------------------------------------------------------------------------------------------
def family():
    Sval = (sp.exp(-2 * p0) * Q2 ** 2 - sp.exp(2 * p0) * Q1 ** 2) / (2 * M)
    Veff_inf = sp.exp(2 * p0) * Q1 ** 2 + sp.exp(-2 * p0) * Q2 ** 2
    Dval = sp.sqrt(M ** 2 + Sval ** 2 - Veff_inf)
    return {S: Sval, Dl: Dval}


def K_of_r(var):
    """K(r) for the variation d/d(var), var in (M, p0, Q1, Q2)."""
    fam = family()
    Af = A.subs(fam); Bf = sp.sqrt(B2.subs(fam)); phf = phi.subs(fam)
    At1 = -Q1 * sp.exp(2 * p0) / (r - S); At2 = -Q2 * sp.exp(-2 * p0) / (r + S)
    At1 = At1.subs(fam); At2 = At2.subs(fam)
    d = lambda X: sp.diff(X, var)
    dQ1 = 1 if var == Q1 else 0
    dQ2 = 1 if var == Q2 else 0
    K = sp.Rational(1, 2) * Bf * (sp.diff(Af, r) * d(Bf) - sp.diff(Bf, r) * d(Af) - 2 * Af * sp.diff(d(Bf), r)) \
        - Af * Bf ** 2 * sp.diff(phf, r) * d(phf) - At1 * dQ1 - At2 * dQ2
    return K


if __name__ == '__main__':
    import random
    random.seed(1)
    num = {M: sp.Rational(3), p0: sp.Rational(1, 5), Q1: sp.Rational(1), Q2: sp.Rational(2)}
    fam = family()
    Sn = fam[S].subs(num); Dn = fam[Dl].subs(num)
    rpn, rmn = (M + Dl).subs(fam).subs(num), (M - Dl).subs(fam).subs(num)
    print('numeric point: Sigma =', sp.N(Sn), ' Delta =', sp.N(Dn), ' r+ =', sp.N(rpn), ' r- =', sp.N(rmn))
    expected = {M: 1, p0: Sn, Q1: 0, Q2: 0}          # K = dM + Sigma dphi_inf (gauge A_t(oo) = 0)
    for var in (M, p0, Q1, Q2):
        K = K_of_r(var)
        Kp = sp.diff(K, r).subs(num)
        vals = [sp.N(Kp.subs(r, rr), 20) for rr in (sp.Rational(7, 2), 5, 11, sp.Rational(5, 2))]
        Kout = sp.N(K.subs(num).subs(r, sp.Rational(7, 2)), 30)
        Kin = sp.N(K.subs(num).subs(r, (rpn + rmn) / 2), 30)
        print('d/d', var, ':  dK/dr at sample radii =', [float(v) for v in vals],
              '  K(r=7/2) =', float(Kout), ' K(inside, r=(r++r-)/2) =', float(Kin))
        ck.small('d/d%s: dK/dr at four radii (inside and outside)' % var, ck.max_abs(vals), 1e-100)
        ck.close('d/d%s: K outside = dM + Sigma dphi_inf' % var, Kout, sp.N(expected[var], 30), 1e-20)
        ck.close('d/d%s: K in region II = K outside' % var, Kin, Kout, 1e-20)
    ck.finish()
