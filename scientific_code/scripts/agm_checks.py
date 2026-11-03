r"""
agm_checks.py -- the connections with Astefanesei, Goldstein and Mahapatra, hep-th/0611140 (AGM), and with
Goldstein and Yavartanoo, arXiv:0711.3374 (GY), quoted in the paper.

(1) Static D = 4, V = 0 solutions: F = a^2 b^2 has F'' = 2, so C = F'^2/4 - F is constant.  At a horizon
    C = 4 T_h^2 S_h^2.  At infinity (asymptotic series, solved symbolically from the reduced equations) C equals
    M^2 + g Sigma^2 - Veff(phi_inf), which is AGM eq. (30).  Hence M^2 + g Sigma^2 - Veff = 4 T_+^2 S_+^2 = 4 T_-^2 S_-^2.
(2) The exact dilaton family of Section 5.5, which is AGM's solution Ia in a two-electric frame:
    (a) AGM (30) at both horizons;
    (b) the shift symmetry phi -> phi + l, A1 -> e^l A1, A2 -> e^-l A2 gives Sigma = Phi2 Q2 - Phi1 Q1 at both horizons;
    (c) hence the dressed first law dM = T dS + Phibar.dQbar (AGM eq. 67) at both horizons;
    (d) Sigma = -dM/dphi_inf at fixed S_+ and charges (AGM eqs. 61-64), and also at fixed S_-;
    (e) linear perturbation theory (AGM eqs. 117-120): (phi_pm - phi_*)/(phi_inf - phi_*) -> P_1(pm 1)/P_1(z_inf).
(3) GY's nonlinear electrodynamics, L = (1/16 pi)[R - L_F(F^2)], L_F = F^2 cosh^-2((lam^2 F^2/2)^(1/4)), with magnetic
    charge P.  The static solutions are a^2 = 1 - 2M/r + (2 M0/r) tanh(k/r), k = sqrt|lam P|, M0 = P^2/(2k); the centre
    is regular iff M = M0, and the extremal black hole is regular iff lam/P = W0(1/e) (GY eq. 2.14).
    (a) (a^2 b^2)'' - 2 = 2 r^2 (s L_F'(s) - L_F(s)), s = F^2 = 2P^2/r^4: the Weyl radius is not harmonic and
        T_+S_+ + T_-S_- != 0, for the regular non-extremal black holes as well, so the hyperbola of the Weyl-radius
        lemma fails, and S_+S_- depends on M at fixed P;
    (b) near-extremal black holes with M slightly above M0 have a singular centre, but the region-II Legendre identity
        I_II = -(T_+S_+ - T_-S_-) holds and I_II/(kappa_+ - kappa_-) -> f, with -2 pi f the entropy of GY's extremal
        regular black hole: the extremal limit of region II does not see the centre.
"""
import sympy as sp
import mpmath as mp
import checklib as ck

mp.mp.dps = 30


# ---------------------------------------------------------------------------------------------------- (1) asymptotics
def asymptotic_check():
    x = sp.Symbol('x', positive=True)                       # x = 1/r
    M, Sig, b0 = sp.symbols('M Sigma b0')
    V0, V1, V2, g0, g1 = sp.symbols('V0 V1 V2 g0 g1')       # Veff and g about phi_inf (single scalar)
    al2, al3, be1, be2, c2, c3 = sp.symbols('alpha2 alpha3 beta1 beta2 c2 c3')
    r = 1 / x
    a2 = 1 - 2 * M * x + al2 * x ** 2 + al3 * x ** 3
    b = r + b0 + be1 * x + be2 * x ** 2                       # b0: the shift freedom of r is left free
    dphi = Sig * x + c2 * x ** 2 + c3 * x ** 3
    Veff = V0 + V1 * dphi + V2 * dphi ** 2 / 2
    g = g0 + g1 * dphi
    d = lambda f: -x ** 2 * sp.diff(f, x)                    # d/dr
    co = lambda E, n: sp.expand(sp.series(E, x, 0, n + 1).removeO()).coeff(x, n)
    E1 = d(d(b)) + g * d(dphi) ** 2 * b                       # b''/b + g phi'^2 = 0, times b
    E3 = sp.Rational(1, 2) * d(b ** 2 * d(a2)) - Veff / b ** 2  # tt component (V = 0)
    E4 = -1 + a2 * d(b) ** 2 + sp.Rational(1, 2) * d(a2) * d(b ** 2) + Veff / b ** 2 - a2 * b ** 2 * g * d(dphi) ** 2
    print('(1) leading asymptotic equations:  (red1) at O(r^-3): %s = 0 ;  (tt) at O(r^-2): %s = 0'
          % (co(E1, 3), co(E3, 2)))
    sol = sp.solve([co(E1, 3), co(E3, 2)], [be1, al2], dict=True)[0]
    cons = [sp.simplify(co(E4, n).subs(sol)) for n in range(3)]
    print('    => beta1 =', sol[be1], ', alpha2 =', sol[al2],
          ';  constraint (red4) at O(r^0), O(r^-1), O(r^-2):', cons)
    ck.true('(1) the constraint (red4) holds at O(r^0), O(r^-1), O(r^-2)', all(c == 0 for c in cons))
    F = sp.expand(a2 * b ** 2)
    C = sp.Rational(1, 4) * d(F) ** 2 - F
    C0 = sp.simplify(co(sp.expand(C.subs(sol)), 0))
    print('    C = F\'^2/4 - F at infinity:', C0, ';  C - (M^2 + g Sigma^2 - Veff(phi_inf)) =',
          sp.simplify(C0 - (M ** 2 + g0 * Sig ** 2 - V0)), '(independent of the shift b0)')
    ck.zero('(1) F\'^2/4 - F at infinity = M^2 + g Sigma^2 - Veff(phi_inf), for any shift b0',
            C0 - (M ** 2 + g0 * Sig ** 2 - V0))


# ------------------------------------------------------------------------------------------- (2) exact dilaton family
def dilaton(M, phinf, Q1, Q2):
    Sig = (mp.e ** (-2 * phinf) * Q2 ** 2 - mp.e ** (2 * phinf) * Q1 ** 2) / (2 * M)
    Vinf = mp.e ** (2 * phinf) * Q1 ** 2 + mp.e ** (-2 * phinf) * Q2 ** 2
    Dl = mp.sqrt(M ** 2 + Sig ** 2 - Vinf)
    out = dict(Sig=Sig, Vinf=Vinf, Delta=Dl)
    for lab, rh, ro in (('+', M + Dl, M - Dl), ('-', M - Dl, M + Dl)):
        S = mp.pi * (rh ** 2 - Sig ** 2)
        T = (rh - ro) / (4 * mp.pi * (rh ** 2 - Sig ** 2))
        Phi1 = mp.e ** (2 * phinf) * Q1 / (rh - Sig)
        Phi2 = mp.e ** (-2 * phinf) * Q2 / (rh + Sig)
        out[lab] = dict(r=rh, S=S, T=T, Phi1=Phi1, Phi2=Phi2, phi=phinf + mp.log((rh + Sig) / (rh - Sig)) / 2)
    return out


def dilaton_checks():
    p0 = [mp.mpf('1.3'), mp.mpf('0.2'), mp.mpf('0.7'), mp.mpf('0.9')]   # M, phi_inf, Q1, Q2
    D = dilaton(*p0)
    print('\n(2) exact dilaton family (AGM solution Ia), M, phi_inf, Q1, Q2 =', [float(v) for v in p0])
    for h in '+-':
        d = D[h]
        e1 = 4 * d['T'] ** 2 * d['S'] ** 2 - (p0[0] ** 2 + D['Sig'] ** 2 - D['Vinf'])
        e2 = D['Sig'] - (d['Phi2'] * p0[3] - d['Phi1'] * p0[2])
        print('    %s horizon: 4 T^2 S^2 - (M^2 + Sigma^2 - Veff(phi_inf)) = %.2e ;  Sigma - (Phi2 Q2 - Phi1 Q1) = %.2e'
              % (h, e1, e2))
        ck.small('(2) AGM (30) at the %s horizon: 4 T^2 S^2 = M^2 + Sigma^2 - Veff(phi_inf)' % h, e1, 1e-25)
        ck.small('(2) shift-symmetry identity at the %s horizon: Sigma = Phi2 Q2 - Phi1 Q1' % h, e2, 1e-25)
    # (c) dressed first law: Qbar1 = e^phi Q1, Qbar2 = e^-phi Q2, Phibar1 = e^-phi Phi1, Phibar2 = e^phi Phi2
    names = ['M', 'phi_inf', 'Q1', 'Q2']
    for h in '+-':
        res = []
        for k in range(4):
            def path(eps, k=k):
                p = list(p0)
                p[k] += eps
                return p
            dM = mp.diff(lambda e: path(e)[0], 0)
            dS = mp.diff(lambda e: dilaton(*path(e))[h]['S'], 0)
            dQb1 = mp.diff(lambda e: mp.e ** path(e)[1] * path(e)[2], 0)
            dQb2 = mp.diff(lambda e: mp.e ** (-path(e)[1]) * path(e)[3], 0)
            d = D[h]
            res.append(dM - d['T'] * dS - mp.e ** (-p0[1]) * d['Phi1'] * dQb1 - mp.e ** p0[1] * d['Phi2'] * dQb2)
        print('    dressed first law at the %s horizon, residuals for d/d(%s): %s'
              % (h, ', '.join(names), ', '.join('%.1e' % abs(v) for v in res)))
        ck.small('(2) AGM (67) dressed first law at the %s horizon, all four variations' % h, ck.max_abs(res), 1e-25)
    # (d) Sigma = -dM/dphi_inf at fixed S_h and charges
    for h in '+-':
        S0 = D[h]['S']
        dSdM = mp.diff(lambda m: dilaton(m, p0[1], p0[2], p0[3])[h]['S'], p0[0])
        dSdp = mp.diff(lambda f: dilaton(p0[0], f, p0[2], p0[3])[h]['S'], p0[1])
        Mf = lambda f: mp.findroot(lambda m: dilaton(m, f, p0[2], p0[3])[h]['S'] - S0, p0[0])
        dMdp = mp.diff(Mf, p0[1])
        print('    fixed S_%s: dM/dphi_inf = %.15f (implicit %.15f) ;  -Sigma = %.15f'
              % (h, dMdp, -dSdp / dSdM, -D['Sig']))
        ck.close('(2) at fixed S_%s and charges: dM/dphi_inf = -Sigma' % h, dMdp, -D['Sig'], 1e-20)
    # (e) linear perturbation theory about the doubly-extremal point
    Q1, Q2, M = p0[2], p0[3], p0[0]
    phis = mp.log(Q2 / Q1) / 2                               # e^{4 phi_*} = Q2^2/Q1^2
    for dl in (mp.mpf('1e-6'), mp.mpf('1e-9')):
        Dd = dilaton(M, phis + dl, Q1, Q2)
        zinf = M / Dd['Delta']
        up, um = (Dd['+']['phi'] - phis) / dl * zinf, (Dd['-']['phi'] - phis) / dl * zinf
        print('    phi_inf - phi_* = %.0e: (phi_+ - phi_*)/(phi_inf - phi_*) * z_inf = %.10f, '
              '(phi_- - phi_*)/(phi_inf - phi_*) * z_inf = %.10f   [P_1(+-1) = +-1]' % (dl, up, um))
        ck.close('(2) linear theory, phi_inf - phi_* = %.0e: horizon values follow P_1(+1) = +1' % dl, up, 1, 10 * dl)
        ck.close('(2) linear theory, phi_inf - phi_* = %.0e: horizon values follow P_1(-1) = -1' % dl, um, -1, 10 * dl)


# ------------------------------------------------------------------------------- (3) Goldstein--Yavartanoo NED model
def gy_symbolic():
    r, P, k, M = sp.symbols('r P k M', positive=True)
    s = sp.Symbol('s', positive=True)
    lam = sp.Symbol('lambda', positive=True)
    LF = s / sp.cosh((lam ** 2 * s / 2) ** sp.Rational(1, 4)) ** 2
    M0 = P ** 2 / (2 * k)
    a2 = 1 - 2 * M / r + 2 * M0 / r * sp.tanh(k / r)
    # field equation: m'(r) = r^2 L_F(2 P^2/r^4)/4 with m = M - M0 tanh(k/r), k = sqrt(lam P)
    m = M - M0 * sp.tanh(k / r)
    resid = sp.simplify((sp.diff(m, r) - r ** 2 * LF.subs(s, 2 * P ** 2 / r ** 4) / 4).subs(lam, k ** 2 / P))
    weyl = sp.simplify(sp.diff(a2 * r ** 2, r, 2) - 2
                       - 2 * r ** 2 * (s * sp.diff(LF, s) - LF).subs(s, 2 * P ** 2 / r ** 4).subs(lam, k ** 2 / P))
    print('\n(3) GY model: m\'(r) - r^2 L_F/4 =', resid, ';  (a^2 b^2)\'\' - 2 - 2 r^2 (s L_F\' - L_F) =', weyl)
    ck.zero('(3) GY solution solves the field equation m\' = r^2 L_F/4', resid)
    ck.zero('(3) GY: (a^2 b^2)\'\' - 2 = 2 b^2 (s L_F\' - L_F)', weyl)


def gy_numeric():
    w0 = mp.lambertw(1 / mp.e).real
    P = mp.mpf(1)
    k = mp.sqrt(w0) * P                                      # lam/P = w0: the extremal black hole is regular
    M0 = P ** 2 / (2 * k)
    a2 = lambda r, M: 1 - 2 * M / r + 2 * M0 / r * mp.tanh(k / r)
    da2 = lambda r, M: mp.diff(lambda x: a2(x, M), r)
    # extremal radius: a2 = da2 = 0 at M = M0
    r0 = mp.findroot(lambda x: da2(x, M0), 0.6)
    print('    lam/P = W0(1/e) = %.12f, M0 = P^2/2k = %.12f; extremal: a^2(r0) = %.1e at r0 = %.12f, '
          'S_ext = pi r0^2 = %.12f, GY (2.19) 4 pi w0 P^2/(1+w0)^2 = %.12f'
          % (w0, M0, a2(r0, M0), r0, mp.pi * r0 ** 2, 4 * mp.pi * w0 * P ** 2 / (1 + w0) ** 2))
    Sext = mp.pi * r0 ** 2
    ck.small('(3) GY extremal horizon: a^2(r0) = 0 at M = M0', a2(r0, M0), 1e-25)
    ck.close('(3) GY extremal entropy = GY (2.19)', Sext, 4 * mp.pi * w0 * P ** 2 / (1 + w0) ** 2, 1e-25)

    def horizons(M):
        """the two largest roots of a^2 (for M > M0 there is a third one, near r = 2(M - M0), below them)"""
        xs = [r0 * (mp.mpf(1) / 5 + mp.mpf(i) / 1000) for i in range(2801)]
        vals = [a2(x, M) for x in xs]
        roots = [mp.findroot(lambda t: a2(t, M), (xs[i], xs[i + 1]), solver='bisect')
                 for i in range(len(xs) - 1) if vals[i] * vals[i + 1] < 0]
        if len(roots) < 2:
            raise ValueError('fewer than two horizons')
        return roots[-1], roots[-2]

    def LF(s):
        lam = k ** 2 / P
        return s / mp.cosh((lam ** 2 * s / 2) ** mp.mpf(0.25)) ** 2

    def ricci(r, M):
        f = lambda x: a2(x, M)
        return -mp.diff(f, r, 2) - 4 * mp.diff(f, r) / r + 2 * (1 - f(r)) / r ** 2

    print('    M/M0      r_+          r_-          T+S+ + T-S-     S+S-        I_II + (T+S+ - T-S-)   '
          'I_II/(k+ - k-)   -2 pi I_II/(k+ - k-)')
    devs = []
    for dM in (mp.mpf('0.05'), mp.mpf('0.01'), mp.mpf('1e-3'), mp.mpf('1e-4'), mp.mpf('1e-5')):
        M = M0 * (1 + dM)
        rp, rm = horizons(M)
        TS = {}
        kap = {}
        for lab, rh in (('+', rp), ('-', rm)):
            kap[lab] = da2(rh, M) / 2
            TS[lab] = kap[lab] / (2 * mp.pi) * mp.pi * rh ** 2
        I2 = mp.quad(lambda r: r ** 2 * (ricci(r, M) - LF(2 * P ** 2 / r ** 4)), [rm, (rm + rp) / 2, rp]) / 4
        ratio = I2 / (kap['+'] - kap['-'])
        print('    1+%-6s  %.9f  %.9f  %+.6e   %.9f   %.1e               %+.9f     %.9f'
              % (mp.nstr(dM, 2), rp, rm, TS['+'] + TS['-'], mp.pi ** 2 * rp ** 2 * rm ** 2,
                 abs(I2 + (TS['+'] - TS['-'])), ratio, -2 * mp.pi * ratio))
        ck.small('(3) GY, M/M0 = 1 + %s: region-II Komar relation I_II = -(T+S+ - T-S-)' % mp.nstr(dM, 2),
                 I2 + (TS['+'] - TS['-']), 1e-25)
        ck.large('(3) GY, M/M0 = 1 + %s: T+S+ + T-S- is nonzero (no harmonic area product)' % mp.nstr(dM, 2),
                 TS['+'] + TS['-'], dM / 2)
        devs.append(abs(-2 * mp.pi * ratio - Sext))
    ck.true('(3) GY: -2 pi I_II/(k+ - k-) approaches S_ext monotonically as M -> M0',
            all(devs[i + 1] < devs[i] for i in range(len(devs) - 1)), 'deviations %s' % ', '.join('%.1e' % d for d in devs))
    ck.small('(3) GY: -2 pi I_II/(k+ - k-) - S_ext at M/M0 = 1 + 1e-5', devs[-1], 2e-4)
    # GY entropy function at the extremum, f = (v1 - v2)/2 - v1 v2 L_F(2P^2/v2^2)/4 on AdS2 x S2
    v2 = r0 ** 2
    v1 = 2 / mp.diff(lambda x: da2(x, M0), r0)               # AdS2 radius^2 = 2/(a^2)''(r0)
    f = (v1 - v2) / 2 - v1 * v2 * LF(2 * P ** 2 / v2 ** 2) / 4
    print('    near horizon: v1 = %.9f, v2 = %.9f, f = %.9f, -2 pi f = %.12f = S_ext' % (v1, v2, f, -2 * mp.pi * f))
    ck.close('(3) GY entropy function at the AdS2 x S2 extremum: -2 pi f = S_ext', -2 * mp.pi * f, Sext, 1e-20)
    nohor = []
    print('    for M < M0 (repulsive singular centre): ', end='')
    for M in (M0 * mp.mpf('0.999'), M0 * mp.mpf('0.99')):
        try:
            rp, rm = horizons(M)
            print('M/M0 = %.3f: real roots found %.6f, %.6f' % (M / M0, rp, rm), end='; ')
        except Exception:
            print('M/M0 = %.3f: no horizon' % (M / M0), end='; ')
            nohor.append(M)
    print()
    ck.true('(3) GY with lam/P = W0(1/e): no horizon for M < M0 (M/M0 = 0.999, 0.99)', len(nohor) == 2)


def gy_regular_nonextremal():
    """non-extremal regular black holes: M = M0, lam/P < W0(1/e)"""
    P = mp.mpf(1)
    ck_rows = []
    print('    regular non-extremal black holes (M = M0): ', end='')
    for lp in ('0.1', '0.2', '0.25'):
        k = mp.sqrt(mp.mpf(lp)) * P
        M0 = P ** 2 / (2 * k)
        a2 = lambda r: 1 - 2 * M0 / r + 2 * M0 / r * mp.tanh(k / r)
        xs = [mp.mpf(i) / 1000 for i in range(1, 6000)]
        roots = [mp.findroot(a2, (xs[i], xs[i + 1]), solver='bisect')
                 for i in range(len(xs) - 1) if a2(xs[i]) * a2(xs[i + 1]) < 0]
        TS = [rh ** 2 * mp.diff(a2, rh) / 4 for rh in roots]
        print('lam/P = %s: r_pm = %.6f, %.6f, T+S+ + T-S- = %.6f' % (lp, roots[-1], roots[-2], TS[-1] + TS[-2]),
              end='; ')
        ck_rows.append((lp, TS[-1] + TS[-2]))
    print()
    for lp, v in ck_rows:
        ck.large('(3) GY regular black hole lam/P = %s: T+S+ + T-S- is nonzero' % lp, v, 1e-2)


if __name__ == '__main__':
    asymptotic_check()
    dilaton_checks()
    gy_symbolic()
    gy_regular_nonextremal()
    gy_numeric()
    ck.finish()
