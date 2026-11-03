r"""
c3_exact.py -- exact third-order obstruction at an odd integer nu, including the metric backreaction.

Background: the doubly-extremal (constant-scalar) solution with b = r, a^2 b^2 = P(r) = (r - R)(r - 1/R), V_* = 1,
in the variable  z = A - B/r,  A = (R^2 + 1)/(R^2 - 1),  B = 2R/(R^2 - 1)  (z(r+) = 1, z(r-) = -1).
The temperature enters only through R = r+ (r+ r- = V_* = 1).  Two-exponential model V_eff = e^{alpha1 phi} + e^{alpha2 phi}
about its minimum, with p = alpha1 alpha2 = -2 nu(nu+1) and s = alpha1 + alpha2:
    V''/V_* = -p,  V'''/V_* = -p s,  V''''/V_* = -p (s^2 - p).
Expansion  phi = phi_* + eps P_nu(z) + eps^2 phi_2 + eps^3 phi_3,  b = r + eps^2 b_2,  eps = phi_+ - phi_*:
    L phi_2 = (V'''/4) P_nu^2,                       L = d_z (z^2 - 1) d_z - nu(nu+1),  phi_2(1) = 0, regular at z = 1;
    b_2'' = -r phi_1'(r)^2,  b_2(R) = 0,  b_2'(R) = -V''/(2 R (R - 1/R))     (b_+ fixed; constraint (red4) at r+);
    L phi_3 = (V'''/2) P_nu phi_2 + (V''''/12) P_nu^3 - 2 nu(nu+1) (b_2/r) P_nu .
For odd nu, c2 = 0 and the coefficient of log(1+z) in phi_3 at z = -1 gives
    c3 = lim (r - r-) phi'/eps^3 = (1/2)(-1)^nu \int_{-1}^{1} P_nu s_3 dz .
Results (exact rational functions; no numerics):
    nu = 3, p = -24:   c3 = 24 (s^2 - 4)/1001  for every R  (zero only at the B2 couplings s = +-2, e.g. (4,-6)),
    nu = 7, p = -112:  c3 = 7056 (109999 s^2 - 5735000)/19255039375  for every R,
The all-temperature statement follows directly after beta = b_2/r:
    beta_zz = -(P_nu')^2, beta(1) = 0, beta_z(1) = -nu(nu+1)/2.
These equations contain no R. The polynomial computation is cross-checked below against the original
independent r/u computation at general R for nu = 3 and two exact temperatures for nu = 7.
Thus at nu = 7 both c2 and c3 vanish at the non-Toda couplings s^2 = 5735000/109999: there the first obstruction, if
any, is at fourth order or beyond.  A universal third-order obstruction for non-Toda couplings is therefore false;
'generically at third order' is the correct statement.
"""
import time
import sympy as sp
import checklib as ck

z, u = sp.symbols('z u')
R = sp.Symbol('R', positive=True)  # physical nonextremal exterior requires R > 1
s = sp.Symbol('s', real=True)


def c3_coefficient(nu, Rval=None):
    Rv = R if Rval is None else sp.nsimplify(Rval)
    lam2 = nu * (nu + 1)
    p = -2 * lam2
    V2, V3, V4 = -p, -p * s, -p * (s ** 2 - p)
    A = (Rv ** 2 + 1) / (Rv ** 2 - 1)
    B = 2 * Rv / (Rv ** 2 - 1)
    P = sp.legendre(nu, z)
    # second order: expand the source in Legendre polynomials (degree <= 2 nu, no resonance for l != nu)
    src2 = sp.expand(V3 / 4 * P ** 2)
    phi2 = 0
    for l in range(0, 2 * nu + 1):
        cl = sp.integrate(sp.expand(src2 * sp.legendre(l, z)), (z, -1, 1)) * sp.Rational(2 * l + 1, 2)
        if cl != 0:
            assert l * (l + 1) != lam2, 'resonant source at second order'
            phi2 += cl * sp.legendre(l, z) / (l * (l + 1) - lam2)
    phi2 = sp.expand(phi2 - phi2.subs(z, 1) * P)
    Lop = lambda f: sp.diff((z ** 2 - 1) * sp.diff(f, z), z) - lam2 * f
    assert sp.expand(Lop(phi2) - src2) == 0
    # metric backreaction in u = 1/r: z = A - B u, d/dr = -u^2 d/du, so b2'' = u^2 (u^2 b2_u)_u = -r phi1'(r)^2,
    # and phi1'(r) = P'(z) dz/dr = P'(z) B u^2, giving (u^2 b2_u)_u = -B^2 u P'(A - B u)^2.
    Pz = sp.diff(P, z)
    g = sp.expand(-B ** 2 * u * Pz.subs(z, A - B * u) ** 2)
    G = sp.integrate(g, u)                                  # u^2 b2_u = G + C1   (G(0) = 0)
    C0, C1 = sp.symbols('C0 C1')
    b2 = sp.integrate(sp.expand(G / u ** 2), u) - C1 / u + C0   # G/u^2 is a polynomial since G = O(u^2)
    # conditions at r = R (u = 1/R): b2 = 0 and db2/dr = -u^2 db2/du = -V2/(2R(R - 1/R))
    eqs = [b2.subs(u, 1 / Rv), (-u ** 2 * sp.diff(b2, u)).subs(u, 1 / Rv) + V2 / (2 * Rv * (Rv - 1 / Rv))]
    sol = sp.solve(eqs, [C0, C1], dict=True)[0]
    b2 = b2.subs(sol)
    beta = sp.expand(sp.cancel((b2 * u).subs(u, (A - z) / B)))   # b2/r as a polynomial in z
    s3 = sp.expand(V3 / 2 * P * phi2 + V4 / 12 * P ** 3 - 2 * lam2 * beta * P)
    I = sp.integrate(sp.expand(P * s3), (z, -1, 1))
    return sp.factor(sp.cancel(I / (2 * (-1) ** nu)))


def temperature_free_data(nu):
    """Exact polynomial projection for a positive odd integer mode.

    The radial change of variables, not agreement at sampled temperatures,
    removes the temperature from beta. The resonant even modes require their
    second-order logarithm and are deliberately outside this function's scope.
    """
    if isinstance(nu, bool) or not isinstance(nu, (int, sp.Integer)) or nu <= 0 or nu % 2 == 0:
        raise ValueError('a positive odd integer nu is required')
    lam2 = sp.Integer(nu * (nu + 1))
    P = sp.legendre(nu, z)
    source2 = lam2 * s * P ** 2 / 2
    phi2 = sp.Integer(0)
    for degree in range(0, 2 * nu + 1, 2):
        mode = sp.legendre(degree, z)
        projection = sp.integrate(sp.expand(source2 * mode), (z, -1, 1)) * sp.Rational(2 * degree + 1, 2)
        phi2 += projection * mode / (degree * (degree + 1) - lam2)
    phi2 = sp.expand(phi2 - phi2.subs(z, 1) * P)
    beta = -sp.integrate(sp.integrate(sp.diff(P, z) ** 2, z), z)
    beta += (-lam2 / 2 - sp.diff(beta, z).subs(z, 1)) * (z - 1)
    beta = sp.expand(beta - beta.subs(z, 1))
    source3 = sp.expand(lam2 * s * P * phi2 + lam2 * (s ** 2 + 2 * lam2) * P ** 3 / 6 - 2 * lam2 * beta * P)
    c3 = sp.factor(sp.integrate(sp.expand(P * source3), (z, -1, 1)) / (2 * (-1) ** nu))
    return dict(phi2=phi2, beta=beta, source3=source3, c3=c3)


def temperature_free_checks():
    rr, AA, BB = sp.symbols('rr A B', positive=True)
    beta_function = sp.Function('beta')
    zr = AA - BB / rr
    ck.zero('all R: b2 = r beta(A-B/r) gives b2_rr = B^2/r^3 beta_zz',
            sp.diff(rr * beta_function(zr), rr, 2)
            - BB ** 2 / rr ** 3 * sp.diff(beta_function(z), z, 2).subs(z, zr))
    arbitrary_mode = sp.Function('P')
    ck.zero('all R: -r phi1_r^2 = -B^2/r^3 P_z^2',
            -rr * sp.diff(arbitrary_mode(zr), rr) ** 2
            + BB ** 2 / rr ** 3 * sp.diff(arbitrary_mode(z), z).subs(z, zr) ** 2)
    lam2 = sp.Symbol('lambda2', positive=True)
    AA_R = (R ** 2 + 1) / (R ** 2 - 1)
    ck.zero('all R: the outer constraint gives beta_z(1) = -lambda2/2',
            (AA_R - 1) * (-lam2 / 2) + lam2 / (R * (R - 1 / R)))
    for nu in (1, 3, 7):
        data = temperature_free_data(nu)
        lam2 = nu * (nu + 1)
        P = sp.legendre(nu, z)
        ck.zero('nu = %d: temperature-free metric equation' % nu,
                sp.diff(data['beta'], z, 2) + sp.diff(P, z) ** 2)
        ck.zero('nu = %d: beta(1) = 0' % nu, data['beta'].subs(z, 1))
        ck.zero('nu = %d: beta_z(1) = -nu(nu+1)/2' % nu,
                sp.diff(data['beta'], z).subs(z, 1) + sp.Rational(lam2, 2))
        ck.zero('nu = %d: second-order scalar equation' % nu,
                sp.diff((z ** 2 - 1) * sp.diff(data['phi2'], z), z) - lam2 * data['phi2']
                - sp.Rational(lam2, 2) * s * P ** 2)
        ck.zero('nu = %d: phi2(1) = 0' % nu, data['phi2'].subs(z, 1))
    for invalid in (0, 2, sp.Rational(3, 2)):
        try:
            temperature_free_data(invalid)
        except ValueError:
            ck.true('negative control: polynomial routine rejects mode %s' % invalid, True)
        else:
            ck.true('negative control: polynomial routine rejects mode %s' % invalid, False)
    data3 = temperature_free_data(3)
    P3 = sp.legendre(3, z)
    no_metric_c3 = sp.integrate(sp.expand(P3 * (data3['source3'] + 24 * data3['beta'] * P3)), (z, -1, 1)) / -2
    ck.true('negative control: omitting metric backreaction changes c3',
            sp.simplify(no_metric_c3 - data3['c3']) != 0)


def second_order_data():
    r"""Appendix B.2: Gaunt integrals int P_n^3 = 2 (n n n; 0 0 0)^2 and the explicit nu = 3 second-order solution."""
    from sympy.physics.wigner import wigner_3j
    vals = {}
    for n in range(1, 7):
        I = sp.integrate(sp.legendre(n, z) ** 3, (z, -1, 1))
        vals[n] = I
        ck.zero('int_{-1}^{1} P_%d^3 dz = 2 (%d %d %d; 0 0 0)^2 = %s' % (n, n, n, n, I), I - 2 * wigner_3j(n, n, n, 0, 0, 0) ** 2)
    ck.true('the integrals vanish for odd n (parity) and are 4/35, 36/1001, 800/46189 for n = 2, 4, 6',
            vals[1] == vals[3] == vals[5] == 0
            and (vals[2], vals[4], vals[6]) == (sp.Rational(4, 35), sp.Rational(36, 1001), sp.Rational(800, 46189)))
    # nu = 3, alpha1 alpha2 = -24: V3/V_* = 24 s, source (V3/4V_*) P_3^2 = 6 s P_3^2, with phi_2(1) = 0
    nu, lam2 = 3, 12
    P3 = sp.legendre(nu, z)
    phi2 = s / 16 * (z ** 2 - 1) * (20 * z ** 4 + 5 * z ** 2 - 1)
    ck.zero('nu = 3: phi_2 = (s/16)(z^2 - 1)(20 z^4 + 5 z^2 - 1) solves L_3 phi_2 = 6 s P_3^2',
            sp.expand(sp.diff((z ** 2 - 1) * sp.diff(phi2, z), z) - lam2 * phi2 - 6 * s * P3 ** 2))
    for endpoint in (-1, 1):
        ck.zero('nu = 3: phi_2(%s) = 0' % endpoint, phi2.subs(z, endpoint))
    ck.zero('nu = 3: phi_2 has no P_3 component', sp.integrate(phi2 * P3, (z, -1, 1)))
    wrong = sp.diff((z ** 2 - 1) * sp.diff(phi2, z), z) - lam2 * phi2 + 6 * s * P3 ** 2
    ck.true('negative control: reversing the second-order source fails', sp.expand(wrong) != 0)


if __name__ == '__main__':
    second_order_data()
    temperature_free_checks()
    t0 = time.time()
    c3_3 = c3_coefficient(3)
    print('nu = 3, alpha1 alpha2 = -24, general R:  c3 =', c3_3, '  (%.1f s)' % (time.time() - t0))
    ck.zero('nu = 3: c3 = 24 (s^2 - 4)/1001, independent of R (i.e. of the temperature)', c3_3 - 24 * (s ** 2 - 4) / 1001)
    ck.zero('nu = 3: independent radial and polynomial derivations agree for all R', c3_3 - temperature_free_data(3)['c3'])
    for sv, lab in ((-2, 'B2 (4,-6)'), (0, 'symmetric (sqrt24,-sqrt24)'), (-5, '(3,-8)')):
        print('   s = %s (%s): c3 = %s' % (sv, lab, c3_3.subs(s, sv)))
    ck.zero('nu = 3: c3 vanishes at the B2 coupling s = -2 (alpha = (4,-6))', c3_3.subs(s, -2))
    t0 = time.time()
    c3_7 = temperature_free_data(7)['c3']
    print('nu = 7, alpha1 alpha2 = -112, every R > 1:  c3 =', c3_7, '  (%.1f s)' % (time.time() - t0))
    target7 = sp.Rational(7056, 19255039375) * (109999 * s ** 2 - 5735000)
    ck.zero('nu = 7: exact c3 formula for every R > 1', c3_7 - target7)
    ck.zero('nu = 7: independent radial computation at R = 2 agrees', c3_coefficient(7, Rval=2) - target7)
    ck.zero('nu = 7: independent radial computation at R = 3/2 agrees', c3_coefficient(7, Rval=sp.Rational(3, 2)) - target7)
    s0 = sp.sqrt(sp.Rational(5735000, 109999))
    a1, a2 = sp.symbols('a1 a2')
    roots = sp.solve([a1 + a2 - s0, a1 * a2 + 112], [a1, a2], dict=True)
    print('   nu = 7 zeros of c3: s^2 = 5735000/109999 = %.6f, couplings (alpha1, alpha2) = %s' % (
        float(s0 ** 2), [(float(r_[a1]), float(r_[a2])) for r_ in roots]))
    ck.zero('nu = 7: c3 vanishes at s^2 = 5735000/109999', c3_7.subs(s, s0))
    ck.true('nu = 7 is not a rank-two Toda value, so these zeros are non-Toda couplings', 7 * 8 not in (2, 6, 12, 30),
            'nu(nu+1) = 56; the Toda values are 2, 6, 12, 30')
    ck.finish()
