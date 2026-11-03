r"""
complex_horizons.py -- first laws at all (real and complex) horizons and the resulting sum rules.

(A) RN-AdS_4 (static, constant scalars):  A = 1 - 2M/r + Q^2/r^2 + r^2/l^2, B = r, A_t = -Q/r.
    K(r) = (1/2)B[A' dB - B' dA - 2 A dB'] - A_t dQ is a rational function of r and equals dM for all r;
    evaluating it at each root r_i of P(r) = r^2 A gives  dM = T_i dS_i + Phi_i dQ  with
    T_i = A'(r_i)/4pi, S_i = pi r_i^2, Phi_i = Q/r_i  (complex for complex roots).
    At fixed Q:  sum_i 1/(T_i S_i) = 0  and  sum_i 1/T_i = 0  (residue theorem) =>  prod S_i, sum S_i M-independent.
(B) Kerr-Newman-AdS_4: four roots of Delta_r = (r^2+a^2)(1+r^2/l^2) - 2 m r + q^2;
    M = m/Xi^2, J = a m/Xi^2, Q = q/Xi, S_i = pi (r_i^2+a^2)/Xi, Omega_i = a(1+r_i^2/l^2)/(r_i^2+a^2),
    Phi_i = q r_i/(r_i^2+a^2), T_i = Delta_r'(r_i)/(4 pi (r_i^2+a^2)).
    We check dM = T_i dS_i + Omega_i dJ + Phi_i dQ at each root, and at fixed (J,Q,l):
    sum_i 1/(T_i S_i) = 0, sum_i 1/T_i = 0, prod S_i = pi^4 l^4 (Q^4 + 4 J^2), sum S_i = -2 pi l^2.
"""
import mpmath as mp
import sympy as sp
import checklib as ck
mp.mp.dps = 40


def rnads(M, Q, l):
    coeffs = [1 / l ** 2, 0, 1, -2 * M, Q ** 2]     # r^4/l^2 + r^2 - 2Mr + Q^2
    roots = mp.polyroots(coeffs, maxsteps=200, extraprec=100)
    out = []
    for ri in roots:
        A1 = 2 * M / ri ** 2 - 2 * Q ** 2 / ri ** 3 + 2 * ri / l ** 2
        out.append(dict(r=ri, T=A1 / (4 * mp.pi), S=mp.pi * ri ** 2, Phi=Q / ri))
    return out


def check_rnads():
    M, Q, l = mp.mpf('0.9'), mp.mpf('0.5'), mp.mpf('1.7')
    H = rnads(M, Q, l)
    print('RN-AdS roots:', [mp.nstr(h['r'], 8) for h in H])
    h = mp.mpf('1e-15')
    # first law at every root for d/dM and d/dQ (track roots by continuity: sort by closeness)
    for name, dM, dQ in (('d/dM', 1, 0), ('d/dQ', 0, 1)):
        Hp = rnads(M + h * dM, Q + h * dQ, l); Hm = rnads(M - h * dM, Q - h * dQ, l)
        for hh in H:
            sp_ = min(Hp, key=lambda x: abs(x['r'] - hh['r'])); sm_ = min(Hm, key=lambda x: abs(x['r'] - hh['r']))
            dS = (sp_['S'] - sm_['S']) / (2 * h)
            res = dM - (hh['T'] * dS + hh['Phi'] * dQ)
            print('   %s at r=%s : dM - (T dS + Phi dQ) = %s' % (name, mp.nstr(hh['r'], 6), mp.nstr(res, 3)))
            ck.small('RN-AdS: first law %s at the root r = %s' % (name, mp.nstr(hh['r'], 6)), res, 1e-22)
    s1, s2 = sum(1 / (x['T'] * x['S']) for x in H), sum(1 / x['T'] for x in H)
    print('   sum 1/(T S) =', mp.nstr(s1, 3), '  sum 1/T =', mp.nstr(s2, 3))
    ck.small('RN-AdS: sum_i 1/(T_i S_i) = 0 over the four roots', s1, 1e-30)
    ck.small('RN-AdS: sum_i 1/T_i = 0 over the four roots', s2, 1e-30)
    ck.large('negative control: RN-AdS sum 1/T over the two real roots only is nonzero',
             sum(1 / x['T'] for x in H if abs(mp.im(x['r'])) < 1e-30), 1e-2)
    pr, sm = mp.fprod(x['S'] for x in H) - mp.pi ** 4 * Q ** 4 * l ** 4, sum(x['S'] for x in H) + 2 * mp.pi * l ** 2
    print('   prod S - pi^4 Q^4 l^4 =', mp.nstr(pr, 3), '   sum S + 2 pi l^2 =', mp.nstr(sm, 3))
    ck.small('RN-AdS: prod_i S_i = pi^4 Q^4 l^4', pr, 1e-30)
    ck.small('RN-AdS: sum_i S_i = -2 pi l^2', sm, 1e-30)
    prods, reals = [], []
    for Mx in (mp.mpf('0.9'), mp.mpf('1.3'), mp.mpf('2.0')):
        Hx = rnads(Mx, Q, l)
        real = [x for x in Hx if abs(mp.im(x['r'])) < 1e-30]
        prods.append(mp.fprod(x['S'] for x in Hx)); reals.append(mp.fprod(x['S'] for x in real))
        print('   M = %s: S+ S- (real roots) = %s ;  prod over all roots = %s' % (
            Mx, mp.nstr(reals[-1], 6), mp.nstr(mp.re(prods[-1]), 12)))
    ck.small('RN-AdS: prod over all roots is the same at M = 0.9, 1.3, 2.0', ck.max_abs(p - prods[0] for p in prods), 1e-30)
    ck.large('RN-AdS: S+S- over the two real roots depends on M (M = 0.9 vs 2.0)', reals[2] - reals[0], 1e-2)


def knads(m, a, q, l):
    Xi = 1 - a ** 2 / l ** 2
    # Delta_r = r^4/l^2 + r^2 (1 + a^2/l^2) - 2 m r + a^2 + q^2
    roots = mp.polyroots([1 / l ** 2, 0, 1 + a ** 2 / l ** 2, -2 * m, a ** 2 + q ** 2], maxsteps=200, extraprec=100)
    out = []
    for ri in roots:
        D1 = 4 * ri ** 3 / l ** 2 + 2 * ri * (1 + a ** 2 / l ** 2) - 2 * m
        W = ri ** 2 + a ** 2
        out.append(dict(r=ri, S=mp.pi * W / Xi, T=D1 / (4 * mp.pi * W), Om=a * (1 + ri ** 2 / l ** 2) / W, Phi=q * ri / W))
    return out, dict(M=m / Xi ** 2, J=a * m / Xi ** 2, Q=q / Xi)


def check_knads():
    m, a, q, l = mp.mpf('1.1'), mp.mpf('0.4'), mp.mpf('0.6'), mp.mpf('2.3')
    H, C = knads(m, a, q, l)
    print('KN-AdS roots:', [mp.nstr(x['r'], 8) for x in H])
    h = mp.mpf('1e-15')
    for i, name in enumerate(['m', 'a', 'q']):
        pp = [m, a, q]; pm = [m, a, q]; pp[i] += h; pm[i] -= h
        Hp, Cp = knads(*pp, l); Hm, Cm = knads(*pm, l)
        dM = (Cp['M'] - Cm['M']) / (2 * h); dJ = (Cp['J'] - Cm['J']) / (2 * h); dQ = (Cp['Q'] - Cm['Q']) / (2 * h)
        for hh in H:
            sp_ = min(Hp, key=lambda x: abs(x['r'] - hh['r'])); sm_ = min(Hm, key=lambda x: abs(x['r'] - hh['r']))
            dS = (sp_['S'] - sm_['S']) / (2 * h)
            res = dM - (hh['T'] * dS + hh['Om'] * dJ + hh['Phi'] * dQ)
            print('   d/d%s at r=%s : first-law residual = %s' % (name, mp.nstr(hh['r'], 6), mp.nstr(res, 3)))
            ck.small('KN-AdS: first law d/d%s at the root r = %s' % (name, mp.nstr(hh['r'], 6)), res, 1e-22)
    s1, s2 = sum(1 / (x['T'] * x['S']) for x in H), sum(1 / x['T'] for x in H)
    print('   sum 1/(T S) =', mp.nstr(s1, 3), '  sum 1/T =', mp.nstr(s2, 3))
    ck.small('KN-AdS: sum_i 1/(T_i S_i) = 0', s1, 1e-30)
    ck.small('KN-AdS: sum_i 1/T_i = 0', s2, 1e-30)
    pr, sm = mp.fprod(x['S'] for x in H) - mp.pi ** 4 * l ** 4 * (C['Q'] ** 4 + 4 * C['J'] ** 2), sum(x['S'] for x in H) + 2 * mp.pi * l ** 2
    print('   prod S - pi^4 l^4 (Q^4 + 4J^2) =', mp.nstr(pr, 3), '   sum S + 2 pi l^2 =', mp.nstr(sm, 3))
    ck.small('KN-AdS: prod_i S_i = pi^4 l^4 (Q^4 + 4 J^2)', pr, 1e-30)
    ck.small('KN-AdS: sum_i S_i = -2 pi l^2', sm, 1e-30)
    # mass-independence at fixed (J, Q): move along the curve of fixed J, Q by solving for (a, q) at new m
    def at_fixed_JQ(mnew):
        J0, Q0 = C['J'], C['Q']
        f = lambda av, qv: [av * mnew / (1 - av ** 2 / l ** 2) ** 2 - J0, qv / (1 - av ** 2 / l ** 2) - Q0]
        av, qv = mp.findroot(f, (a, q))
        return knads(mnew, av, qv, l)
    print('   at m=%s (start): prod S = %s, sum S = %s, S+ S- (real pair) = %s' % (
        m, mp.nstr(mp.fprod(x['S'] for x in H), 12), mp.nstr(sum(x['S'] for x in H), 12),
        mp.nstr(mp.fprod(x['S'] for x in H if abs(mp.im(x['r'])) < 1e-30), 12)))
    P0, S0, R0 = mp.fprod(x['S'] for x in H), sum(x['S'] for x in H), mp.fprod(x['S'] for x in H if abs(mp.im(x['r'])) < 1e-30)
    for mnew in (mp.mpf('1.3'), mp.mpf('2.0')):
        H2, C2 = at_fixed_JQ(mnew)
        ck.small('KN-AdS at m = %s, same (J, Q): prod S unchanged' % mnew, mp.fprod(x['S'] for x in H2) - P0, 1e-25)
        ck.small('KN-AdS at m = %s, same (J, Q): sum S unchanged' % mnew, sum(x['S'] for x in H2) - S0, 1e-25)
        ck.large('KN-AdS at m = %s, same (J, Q): S+S- of the real pair changes' % mnew,
                 mp.fprod(x['S'] for x in H2 if abs(mp.im(x['r'])) < 1e-30) - R0, 1e-2)
        print('   at m=%s (same J,Q): prod S = %s, sum S = %s, S+ S- (real pair) = %s' % (
            mnew, mp.nstr(mp.fprod(x['S'] for x in H2), 12), mp.nstr(sum(x['S'] for x in H2), 12),
            mp.nstr(mp.fprod(x['S'] for x in H2 if abs(mp.im(x['r'])) < 1e-30), 12)))


def K_rnads_symbolic():
    """K(r) of Prop. 3.1 for RN-AdS (b = r, A_t = -Q/r): identically dM for all complex r."""
    import sympy as sp
    r, M, Q, l, dM, dQ = sp.symbols('r M Q l dM dQ')
    a2 = 1 - 2 * M / r + Q ** 2 / r ** 2 + r ** 2 / l ** 2
    da2 = sp.diff(a2, M) * dM + sp.diff(a2, Q) * dQ
    b, db = r, 0
    At = -Q / r
    K = sp.Rational(1, 2) * b * (sp.diff(a2, r) * db - sp.diff(b, r) * da2 - 2 * a2 * sp.diff(db, r)) - At * dQ
    return sp.simplify(K)


def ddim_sum_rules(D=5):
    """RN-AdS_D with constant scalars: P(r) = r^{2(D-3)} a^2; check sum_i 1/(T_i S_i) = sum_i 1/T_i = 0 over all roots."""
    M_, q, l = mp.mpf('0.7'), mp.mpf('0.3'), mp.mpf('1.4')
    n = 2 * (D - 2)
    # P(r) = r^{2(D-2)}/l^2 + r^{2(D-3)} - mu r^{D-3} + q^2
    coeffs = [mp.mpf(0)] * (n + 1)
    coeffs[0] = 1 / l ** 2                       # r^{2(D-2)}
    coeffs[n - 2 * (D - 3)] += 1                 # r^{2(D-3)}
    coeffs[n - (D - 3)] += -M_                   # r^{D-3}
    coeffs[n] += q ** 2
    roots = mp.polyroots(coeffs, maxsteps=400, extraprec=200)
    dP = lambda x: sum((n - k) * coeffs[k] * x ** (n - k - 1) for k in range(n))
    Om = 2 * mp.pi ** (mp.mpf(D - 1) / 2) / mp.gamma(mp.mpf(D - 1) / 2)
    T = [dP(x) / (4 * mp.pi * x ** (2 * (D - 3))) for x in roots]
    S = [Om * x ** (D - 2) / 4 for x in roots]
    return roots, sum(1 / (t * s) for t, s in zip(T, S)), sum(1 / t for t in T)


if __name__ == '__main__':
    Kr = K_rnads_symbolic()
    print('K(r) for RN-AdS from the radial Noether function of Prop. 3.1 (b = r, A_t = -Q/r):', Kr)
    ck.zero('RN-AdS: K(r) = dM identically in r', Kr - sp.Symbol('dM'))
    check_rnads()
    check_knads()
    for D in (5, 6):
        roots, s1, s2 = ddim_sum_rules(D)
        print('D=%d RN-AdS: %d roots; sum 1/(T S) = %s ; sum 1/T = %s' % (D, len(roots), mp.nstr(s1, 3), mp.nstr(s2, 3)))
        ck.small('D=%d RN-AdS: sum_i 1/(T_i S_i) = 0 over all %d roots' % (D, len(roots)), s1, 1e-30)
        ck.small('D=%d RN-AdS: sum_i 1/T_i = 0 over all %d roots' % (D, len(roots)), s2, 1e-30)
    ck.finish()
