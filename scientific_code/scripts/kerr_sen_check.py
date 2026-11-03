r"""
kerr_sen_check.py -- rotating test of the inner-outer identity with non-trivial scalars and moduli.
Kerr-Sen in the axion-dilaton theory (see kerr_sen_fields.py: c1=1, c2=1, kappa=-1), with the asymptotic
dilaton chi_inf = c restored by the SL(2,R) scaling  chi -> chi + c, xi -> e^{-2c} xi, A -> e^{c} A.
Physical charge Q = (1/4pi) \oint G = e^{-c} q_seed.  Parameters: (M, a, Q, c).
Checks: K_t = dM + Sigma_chi dc, K_phi = -dJ at all radii (incl. region II); first laws at r+ and r-;
T+S+ + T-S- = 0; S+S- = 4 pi^2 J^2 independent of the modulus c while S+ and S- separately depend on it.
"""
import sympy as sp
import mpmath as mp
from stationary_k_num import make_k, K_of_r, r, th
from eom_num import make_residuals
import checklib as ck

M, a, Q, c = sp.symbols('M a Q c', real=True)
qs = sp.exp(c) * Q                      # seed charge
r2 = qs ** 2 / M
St = r * (r + r2) + a ** 2 * sp.cos(th) ** 2
Dl = r * (r + r2) - 2 * M * r + a ** 2
g = sp.zeros(4, 4)
g[0, 0] = -(1 - 2 * M * r / St)
g[0, 3] = g[3, 0] = -2 * a * M * r * sp.sin(th) ** 2 / St
g[3, 3] = sp.sin(th) ** 2 * (r * (r + r2) + a ** 2 + 2 * M * r * a ** 2 * sp.sin(th) ** 2 / St)
g[1, 1] = St / Dl
g[2, 2] = St
chi = c + sp.log((r ** 2 + a ** 2 * sp.cos(th) ** 2) / St) / 2
xi = sp.exp(-2 * c) * (qs ** 2 / M) * a * sp.cos(th) / (r ** 2 + a ** 2 * sp.cos(th) ** 2)
Avec = [sp.exp(c) * (-qs * r / St), sp.Integer(0), sp.Integer(0), sp.exp(c) * qs * r * a * sp.sin(th) ** 2 / St]
scal = [chi, xi]
gsig = lambda s: sp.diag(1, sp.exp(4 * s[0]) / 4)
fcoup = lambda s: sp.exp(-2 * s[0])
ftcoup = lambda s: -s[1]
params = (M, a, Q, c)

if __name__ == '__main__':
    mp.mp.dps = 20
    pv = (mp.mpf('1.4'), mp.mpf('0.5'), mp.mpf('0.8'), mp.mpf('0.3'))
    # field equations with the modulus switched on
    res = make_residuals(g, Avec, scal, gsig, fcoup, ftcoup, lambda s: sp.Integer(0), params)
    E, Mx, S = res(mp.mpf('2.9'), mp.mpf('1.1'), pv)
    print('field equations: max residual =', mp.nstr(ck.max_abs((E, Mx, S)), 3))
    ck.small('Kerr-Sen field equations with the modulus switched on', ck.max_abs((E, Mx, S)), 1e-18)
    Mn, an, Qn, cn = pv
    r2n = mp.e ** (2 * cn) * Qn ** 2 / Mn
    sq = mp.sqrt((Mn - r2n / 2) ** 2 - an ** 2)
    rp, rm = Mn - r2n / 2 + sq, Mn - r2n / 2 - sq
    print('r+ =', mp.nstr(rp, 10), ' r- =', mp.nstr(rm, 10))
    radii = [mp.mpf(30), mp.mpf(4), (rp + rm) / 2, rm / 2]
    Kinf = {}
    expK = {('M', 't'): 1, ('M', 'phi'): -pv[1], ('a', 't'): 0, ('a', 'phi'): -pv[0], ('Q', 't'): 0, ('Q', 'phi'): 0,
            ('c', 't'): -mp.e ** (2 * pv[3]) * pv[2] ** 2 / pv[0] / 2, ('c', 'phi'): 0}
    for var in params:
        h = g.diff(var)
        dA = [x.diff(var) for x in Avec]
        ds = [x.diff(var) for x in scal]
        kf = make_k(g, Avec, scal, gsig, fcoup, ftcoup, h, dA, ds, params)
        for name, xv in (('t', [1, 0, 0, 0]), ('phi', [0, 0, 0, 1])):
            vals = [K_of_r(kf, rr, pv, xv) for rr in radii]
            Kinf[(str(var), name)] = vals[0]
            print('  d/d%s  K_%-3s =' % (var, name), [mp.nstr(v, 12) for v in vals])
            ck.small('d/d%s K_%s at r = 30, 4, (r+ + r-)/2, r-/2 equals %s' % (var, name, mp.nstr(expK[(str(var), name)], 8)),
                     ck.max_abs(v - expK[(str(var), name)] for v in vals), 1e-19)
    print('  expected: K_t(d/dc) = Sigma_chi = -r2/2 =', mp.nstr(-r2n / 2, 12), ';  K_phi = -dJ: d/dM -> -a, d/da -> -M')
    def hor(Mv, av, Qv, cv, lab):
        r2v = mp.e ** (2 * cv) * Qv ** 2 / Mv
        s = mp.sqrt((Mv - r2v / 2) ** 2 - av ** 2)
        rh = Mv - r2v / 2 + (s if lab == '+' else -s)
        ro = Mv - r2v / 2 - (s if lab == '+' else -s)
        W = rh * (rh + r2v) + av ** 2
        Phi = mp.e ** cv * (mp.e ** cv * Qv) * rh / W      # -xi_h.A at the horizon (uses Omega_h = a/W)
        return dict(S=mp.pi * W, T=(rh - ro) / (4 * mp.pi * W), Om=av / W, Phi=Phi)
    Hp, Hm = hor(*pv, '+'), hor(*pv, '-')
    print('  T+S+ + T-S- =', mp.nstr(Hp['T'] * Hp['S'] + Hm['T'] * Hm['S'], 5),
          ';  S+S- - 4pi^2 J^2 =', mp.nstr(Hp['S'] * Hm['S'] - 4 * mp.pi ** 2 * (Mn * an) ** 2, 5))
    ck.small('T+S+ + T-S- = 0', Hp['T'] * Hp['S'] + Hm['T'] * Hm['S'], 1e-18)
    ck.small('S+S- = 4 pi^2 J^2', Hp['S'] * Hm['S'] - 4 * mp.pi ** 2 * (Mn * an) ** 2, 1e-18)
    dSc = {}
    eps = mp.mpf('1e-8')
    for i, var in enumerate(['M', 'a', 'Q', 'c']):
        pp = list(pv); pm = list(pv); pp[i] += eps; pm[i] -= eps
        for lab, H in (('+', Hp), ('-', Hm)):
            dS = (hor(*pp, lab)['S'] - hor(*pm, lab)['S']) / (2 * eps)
            lhs = Kinf[(var, 't')] + H['Om'] * Kinf[(var, 'phi')]
            rhs = H['T'] * dS + H['Phi'] * (1 if var == 'Q' else 0)
            print('  first law at r%s, d/d%s: K_xi - (T dS + Phi dQ) =' % (lab, var), mp.nstr(lhs - rhs, 5),
                  '   (dS/d%s = %s)' % (var, mp.nstr(dS, 8)))
            ck.small('first law at r%s, d/d%s (central differences, step 1e-8)' % (lab, var), lhs - rhs, 1e-11)
            if var == 'c':
                dSc[lab] = dS
    ck.large('the modulus moves S+ at fixed (M, a, Q)', dSc['+'], 1e-2)
    ck.large('the modulus moves S- at fixed (M, a, Q)', dSc['-'], 1e-2)
    ck.small('but not S+S-: d(S+S-)/dc = S- dS+/dc + S+ dS-/dc = 0', Hm['S'] * dSc['+'] + Hp['S'] * dSc['-'], 1e-10)
    ck.finish()
