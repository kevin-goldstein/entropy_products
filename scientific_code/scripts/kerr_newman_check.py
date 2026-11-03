r"""
kerr_newman_check.py -- full covariant-phase-space check on Kerr-Newman (G=1, L=(R-F^2)/16pi).
For the variations d/dM, d/da, d/dQ we compute K_xi(r) = \oint_{S_r} k_xi for xi = d_t, d_phi at radii
outside, between and inside the horizons and verify r-independence, K_t = dM, K_phi = -dJ, the first laws
at both horizons with xi_pm = d_t + Omega_pm d_phi (signed T_-), T+S+ + T-S- = 0 and S+S- = pi^2(Q^4+4J^2).
"""
import itertools
import sympy as sp
import mpmath as mp
import checklib as ck
from stationary_k_num import make_k, K_of_r, r, th

mp.mp.dps = 20
M, a, Q = sp.symbols('M a Q', positive=True)
rho2 = r ** 2 + a ** 2 * sp.cos(th) ** 2
Dl = r ** 2 - 2 * M * r + a ** 2 + Q ** 2
g = sp.zeros(4, 4)
g[0, 0] = -(Dl - a ** 2 * sp.sin(th) ** 2) / rho2
g[0, 3] = g[3, 0] = -a * sp.sin(th) ** 2 * (r ** 2 + a ** 2 - Dl) / rho2
g[3, 3] = ((r ** 2 + a ** 2) ** 2 - Dl * a ** 2 * sp.sin(th) ** 2) / rho2 * sp.sin(th) ** 2
g[1, 1] = rho2 / Dl
g[2, 2] = rho2
Avec = [-Q * r / rho2, sp.Integer(0), sp.Integer(0), Q * r * a * sp.sin(th) ** 2 / rho2]
params = (M, a, Q)


def maxwell_split():
    r"""Section 2.4: for fixed xi the Maxwell part of k_xi is  -(1/4pi)[(xi.A) delta G + delta A ^ (xi.G)],  G = *F.
    For delta = d/dQ we build the two terms symbolically.  With xi = d_t the second term does not vanish on a generic
    sphere of the rotating solution (at a = 0.6, Q = 0.7, r = 4 it equals
    (a^2 Q r/2) int_0^pi (r^2 - a^2 cos^2) sin^3/(r^2 + a^2 cos^2)^3 dth = 0.0025787528...), and it decays at infinity.
    With xi = xi_+ = d_t + Omega_+ d_phi, (xi_+.G) pulls back to zero on the horizon and xi_+.A = -Phi_+ there."""
    X = [sp.Symbol('t'), r, th, sp.Symbol('phi')]
    F = sp.Matrix(4, 4, lambda m, n: sp.diff(Avec[n], X[m]) - sp.diff(Avec[m], X[n]))
    gi = sp.simplify(g.inv())
    sg = rho2 * sp.sin(th)                                  # sqrt(-g) for 0 < theta < pi
    Fup = (gi * F * gi).applyfunc(sp.simplify)
    def G(m, n):                    # (*F)_{mn} = (1/2) sqrt(-g) eps_{mnab} F^{ab},  eps_{t r th ph} = +1
        tot = 0
        for p in itertools.permutations(range(4)):
            if p[0] == m and p[1] == n:
                tot += sp.combinatorics.Permutation(list(p)).signature() * Fup[p[2], p[3]]
        return sp.simplify(sg * tot / 2)
    return dict(sg_ok=sp.simplify(sg ** 2 + g.det()), G23=G(2, 3), Gt2=G(0, 2), G32=G(3, 2), Gt3=G(0, 3),
                dA=[sp.diff(c, Q) for c in Avec])


if __name__ == '__main__':
    pv = (mp.mpf('1.3'), mp.mpf('0.6'), mp.mpf('0.7'))
    Mn, an, Qn = pv
    sq = mp.sqrt(Mn ** 2 - an ** 2 - Qn ** 2)
    rp, rm = Mn + sq, Mn - sq
    radii = [mp.mpf(30), mp.mpf(4), (rp + rm) / 2, rm / 2]
    print('r+ =', mp.nstr(rp, 10), ' r- =', mp.nstr(rm, 10), '  radii:', [mp.nstr(x, 4) for x in radii])
    Kinf = {}
    for var in params:
        h = g.diff(var)
        dA = [c.diff(var) for c in Avec]
        kf = make_k(g, Avec, [], lambda s: sp.zeros(0, 0), lambda s: sp.Integer(1), lambda s: sp.Integer(0), h, dA, [], params)
        for name, xi in (('t', [1, 0, 0, 0]), ('phi', [0, 0, 0, 1])):
            vals = [K_of_r(kf, rr, pv, xi) for rr in radii]
            Kinf[(str(var), name)] = vals[0]
            print('  d/d%s  K_%-3s =' % (var, name), [mp.nstr(v, 12) for v in vals])
            exp_ = {('M', 't'): 1, ('a', 't'): 0, ('Q', 't'): 0, ('M', 'phi'): -an, ('a', 'phi'): -Mn, ('Q', 'phi'): 0}
            ck.small('d/d%s K_%s at r = 30, 4, (r+ + r-)/2, r-/2 equals %s' % (var, name, mp.nstr(exp_[(str(var), name)], 6)),
                     ck.max_abs(v - exp_[(str(var), name)] for v in vals), 1e-19)
    print('  expected: K_t = dM -> (1,0,0);  K_phi = -dJ = -(a, M, 0) = ', (-an, -Mn, 0))
    def hor(rh):
        return dict(S=mp.pi * (rh ** 2 + an ** 2), Om=an / (rh ** 2 + an ** 2), Phi=Qn * rh / (rh ** 2 + an ** 2),
                    T=(2 * rh - rp - rm) / (4 * mp.pi * (rh ** 2 + an ** 2)))
    Hp, Hm = hor(rp), hor(rm)
    print('  T+S+ + T-S- =', mp.nstr(Hp['T'] * Hp['S'] + Hm['T'] * Hm['S'], 5),
          ';  S+S- - pi^2(Q^4+4J^2) =', mp.nstr(Hp['S'] * Hm['S'] - mp.pi ** 2 * (Qn ** 4 + 4 * (Mn * an) ** 2), 5))
    ck.small('T+S+ + T-S- = 0', Hp['T'] * Hp['S'] + Hm['T'] * Hm['S'], 1e-18)
    ck.small('S+S- = pi^2 (Q^4 + 4 J^2)', Hp['S'] * Hm['S'] - mp.pi ** 2 * (Qn ** 4 + 4 * (Mn * an) ** 2), 1e-18)
    eps = mp.mpf('1e-8')
    def Sfun(p, lab):
        s = mp.sqrt(p[0] ** 2 - p[1] ** 2 - p[2] ** 2)
        rr = p[0] + s if lab == '+' else p[0] - s
        return mp.pi * (rr ** 2 + p[1] ** 2)
    for i, var in enumerate(['M', 'a', 'Q']):
        pp = list(pv); pm = list(pv); pp[i] += eps; pm[i] -= eps
        for lab, H in (('+', Hp), ('-', Hm)):
            dS = (Sfun(pp, lab) - Sfun(pm, lab)) / (2 * eps)
            lhs = Kinf[(var, 't')] + H['Om'] * Kinf[(var, 'phi')]
            rhs = H['T'] * dS + H['Phi'] * (1 if var == 'Q' else 0)
            print('  first law at r%s, d/d%s: K_{xi} - (T dS + Phi dQ) =' % (lab, var), mp.nstr(lhs - rhs, 5))
            ck.small('first law at r%s, d/d%s (central differences, step 1e-8)' % (lab, var), lhs - rhs, 1e-11)
    # ------------------------------------------------------------------ the two Maxwell terms of k_xi (Section 2.4)
    print('  Maxwell part of k_xi = -(1/4pi)[(xi.A) dG + dA ^ (xi.G)] for d/dQ, G = *F:')
    S = maxwell_split()
    ck.zero('sqrt(-g) = (r^2 + a^2 cos^2) sin(theta)', S['sg_ok'])
    print('     G_thph =', S['G23'], ';  (d_t.G)_th = G_tth =', S['Gt2'])
    dA = S['dA']
    term1_t = Avec[0] * sp.diff(S['G23'], Q)                      # (xi.A) dG_{th ph},   xi = d_t
    term2_t = dA[2] * S['Gt3'] - dA[3] * S['Gt2']                 # (dA ^ xi.G)_{th ph}, xi = d_t
    lam = lambda e: sp.lambdify((r, th) + params, e, 'mpmath')
    f1, f2 = lam(term1_t), lam(term2_t)
    I = lambda f, rv: -2 * mp.pi * mp.quad(lambda tv: f(rv, tv, *pv), [0, mp.pi / 2, mp.pi]) / (4 * mp.pi)
    h = g.diff(Q)
    kfQ = make_k(g, Avec, [], lambda s: sp.zeros(0, 0), lambda s: sp.Integer(1), lambda s: sp.Integer(0),
                 h, [c.diff(Q) for c in Avec], [], params)
    for rr in (mp.mpf(4), mp.mpf(30)):
        I1, I2 = I(f1, rr), I(f2, rr)
        KA = K_of_r(kfQ, rr, pv, [1, 0, 0, 0], split=True)[2]
        print('     xi = d_t, r = %s:  -(1/4pi) oint (xi.A) dG = %s ;  -(1/4pi) oint dA ^ (xi.G) = %s ;  '
              'Maxwell part of K from stationary_k_num = %s' % (mp.nstr(rr, 3), mp.nstr(I1, 15), mp.nstr(I2, 15), mp.nstr(KA, 15)))
        ck.close('xi = d_t, r = %s: the two terms add up to the Maxwell part of K' % mp.nstr(rr, 3), I1 + I2, KA, 1e-18)
        if rr == 4:
            aud = (an ** 2 * Qn * rr / 2) * mp.quad(lambda x: (rr ** 2 - an ** 2 * mp.cos(x) ** 2) * mp.sin(x) ** 3
                                                    / (rr ** 2 + an ** 2 * mp.cos(x) ** 2) ** 3, [0, mp.pi])
            ck.close('xi = d_t, r = 4: dA ^ (xi.G) term = (a^2 Q r/2) int (r^2 - a^2 c^2) s^3/(r^2 + a^2 c^2)^3', I2, aud, 1e-18)
            ck.large('xi = d_t, r = 4: the dA ^ (xi.G) term is nonzero on a sphere of Kerr-Newman', I2, 1e-3)
            I2_4 = I2
            fxA = lam(Avec[0])
            xa = [fxA(rr, tv, *pv) for tv in (mp.mpf('0.1'), mp.mpf('1.5'))]
            ck.large('xi = d_t, r = 4: xi.A depends on theta, so it cannot be taken outside the integral', xa[1] - xa[0], 1e-3)
        else:
            ck.true('xi = d_t: the dA ^ (xi.G) term decays like r^-3 (r = 30 vs r = 4)', abs(I2) < 2 * abs(I2_4) * (4 / rr) ** 3,
                    'ratio %.2e' % abs(I2 / I2_4))
    # horizon generator xi_+ = d_t + Omega_+ d_phi at r = r+ (exact, with M eliminated in favour of r+)
    rps = sp.Symbol('r_p', positive=True)
    onh = lambda e: sp.simplify(e.subs(M, (rps ** 2 + a ** 2 + Q ** 2) / (2 * rps)).subs(r, rps))
    Omp = a / (rps ** 2 + a ** 2)
    ck.zero('xi_+: (xi_+.G)_theta = 0 on the horizon r = r+, so dA ^ (xi_+.G) pulls back to zero there',
            onh(S['Gt2'] + Omp * S['G32']))
    ck.zero('xi_+: xi_+.A = -Phi_+ = -Q r+/(r+^2 + a^2) on the horizon, independent of theta',
            onh(Avec[0] + Omp * Avec[3]) + Q * rps / (rps ** 2 + a ** 2))
    dG = lam(sp.diff(S['G23'], Q))
    for rr in (rp, mp.mpf(4)):
        ck.close('oint dG = 4 pi dQ at r = %s' % mp.nstr(rr, 6), 2 * mp.pi * mp.quad(lambda tv: dG(rr, tv, *pv), [0, mp.pi]), 4 * mp.pi, 1e-18)
    print('     => at the horizon the Maxwell part of k_{xi_+} is -(xi_+.A) dQ = Phi_+ dQ; on a sphere with xi = d_t it is not -(xi.A) dQ')
    ck.finish()
