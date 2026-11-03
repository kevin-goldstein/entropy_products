r"""
complex_horizons_symbolic.py -- the first laws at all (real and complex) horizons and the residue sum rules of
Sections 6.3 and 7.6, as exact identities for general parameters (complex_horizons.py checks the same statements
numerically at sample points, root by root).

A root r_i of a polynomial P depends on the parameters through P(r_i) = 0, so delta r_i = -delta P(r_i)/P'(r_i).
A first law at "every root" is then a rational function of r_i that must vanish modulo P(r_i) = 0: we reduce its
numerator modulo P and check that the remainder is zero.  The sum rules sum_i g(r_i)/P'(r_i) = 0 hold whenever
deg g <= deg P - 2, i.e. when the residue of g/P at infinity vanishes; the invariants follow from Vieta.
(A) RN-AdS_4:  P = r^4/l^2 + r^2 - 2 M r + Q^2,  T_i = P'(r_i)/(4 pi r_i^2), S_i = pi r_i^2, Phi_i = Q/r_i.
(B) Kerr-Newman-AdS_4:  Delta_r = (r^2 + a^2)(1 + r^2/l^2) - 2 m r + q^2, Xi = 1 - a^2/l^2, M = m/Xi^2, J = a m/Xi^2,
    Q = q/Xi, S_i = pi (r_i^2 + a^2)/Xi, Omega_i = a (1 + r_i^2/l^2)/(r_i^2 + a^2), Phi_i = q r_i/(r_i^2 + a^2),
    T_i = Delta_r'(r_i)/(4 pi (r_i^2 + a^2)).
(C) RN-AdS_D, D = 5, 6: P = r^{2(D-2)}/l^2 + r^{2(D-3)} - mu r^{D-3} + q^2.
Scope: fixed cosmological scale l; distinct nonzero roots; Xi != 0 and r_i^2 + a^2 != 0.
The identities are algebraic continuations of the displayed thermodynamic formulae. They do not construct a
complex spacetime, establish regularity of hairy complex roots, or independently derive the rotating CPS form.
The all-root sums are not identities for the pair of real horizons alone. Exceptional parameter values at
which denominators or the discriminant vanish are excluded. Quotient-ring reduction checks these conditions
generically, and explicitly refuses a denominator that has a common polynomial factor with P.
Run:  python3 complex_horizons_symbolic.py   (a few seconds)
"""
import sympy as sp
import checklib as ck

r, w = sp.symbols('r w')
M, Q, l, m, a, q, mu = sp.symbols('M Q l m a q mu', positive=True)


def mod_P(expr, P):
    """Return the numerator remainder in the generic coefficient field.

    For a square-free P, a zero remainder is equivalent to vanishing at every
    root only if the rational expression is defined there. Reject a common
    denominator factor, rather than silently treating a pole as a first law.
    Parameter-dependent exceptional loci remain excluded as stated above.
    """
    num, den = sp.fraction(sp.cancel(expr))
    poly = sp.Poly(P, r)
    if sp.gcd(poly, poly.diff()).degree() != 0:
        raise ValueError('root polynomial must be square-free')
    if sp.gcd(poly, sp.Poly(den, r)).degree() != 0:
        raise ValueError('expression has a denominator vanishing at a horizon')
    return sp.rem(sp.Poly(num, r), poly).as_expr()


def res_inf(f):
    """residue of f(r) at r = infinity, -Res_{w=0} f(1/w)/w^2"""
    g = sp.simplify(f.subs(r, 1 / w) / w ** 2)
    return -sp.residue(g, w, 0)


def droot(P, X):
    """d r_i / dX at a simple root of P"""
    return -sp.diff(P, X) / sp.diff(P, r)


# ------------------------------------------------------------------------------------------------- (A) RN-AdS_4
print('--- (A) RN-AdS_4, general M, Q, l')
P = r ** 4 / l ** 2 + r ** 2 - 2 * M * r + Q ** 2
T = sp.diff(P, r) / (4 * sp.pi * r ** 2)
S = sp.pi * r ** 2
Phi = Q / r
for X, dM, dQ in ((M, 1, 0), (Q, 0, 1)):
    first = T * sp.diff(S, r) * droot(P, X) + Phi * dQ - dM
    ck.zero('RN-AdS: dM = T_i dS_i + Phi_i dQ at every root, for d/d%s (remainder mod P)' % X, mod_P(first, P))
ck.zero('RN-AdS: T_i S_i = P\'(r_i)/4', sp.simplify(T * S - sp.diff(P, r) / 4))
ck.zero('RN-AdS: sum_i 1/(T_i S_i) = 4 sum_i 1/P\'(r_i) = -4 Res_inf 1/P = 0', res_inf(4 / P))
ck.zero('RN-AdS: sum_i 1/T_i = 4 pi sum_i r_i^2/P\'(r_i) = -4 pi Res_inf r^2/P = 0', res_inf(4 * sp.pi * r ** 2 / P))
ck.large('negative control: Res_inf r^3/P != 0 (deg r^3 = deg P - 1)', res_inf(r ** 3 / P).subs({M: 1, Q: 1, l: 1}), 1e-3)
Pm = sp.Poly(sp.expand(l ** 2 * P), r)                    # monic: r^4 + l^2 r^2 - 2 M l^2 r + Q^2 l^2
c = Pm.all_coeffs()
e1, e2, e4 = -c[1], c[2], c[4]
ck.zero('RN-AdS: prod_i S_i = pi^4 (prod_i r_i)^2 = pi^4 Q^4 l^4 (Vieta)', sp.pi ** 4 * e4 ** 2 - sp.pi ** 4 * Q ** 4 * l ** 4)
ck.zero('RN-AdS: sum_i S_i = pi (e_1^2 - 2 e_2) = -2 pi l^2 (Vieta)', sp.pi * (e1 ** 2 - 2 * e2) + 2 * sp.pi * l ** 2)

# ------------------------------------------------------------------------------------------------- (B) Kerr-Newman-AdS_4
print('--- (B) Kerr-Newman-AdS_4, general m, a, q, l')
Dr = (r ** 2 + a ** 2) * (1 + r ** 2 / l ** 2) - 2 * m * r + q ** 2
Xi = 1 - a ** 2 / l ** 2
Mt, Jt, Qt = m / Xi ** 2, a * m / Xi ** 2, q / Xi
W = r ** 2 + a ** 2
Si = sp.pi * W / Xi
Om = a * (1 + r ** 2 / l ** 2) / W
Ph = q * r / W
Ti = sp.diff(Dr, r) / (4 * sp.pi * W)
for X in (m, a, q):
    dS = sp.diff(Si, X) + sp.diff(Si, r) * droot(Dr, X)        # total variation of S_i, root displacement included
    first = Ti * dS + Om * sp.diff(Jt, X) + Ph * sp.diff(Qt, X) - sp.diff(Mt, X)
    ck.zero('KN-AdS: dM = T_i dS_i + Omega_i dJ + Phi_i dQ at every root, for d/d%s (remainder mod Delta_r)' % X, mod_P(first, Dr))
ck.zero('KN-AdS: T_i S_i = Delta_r\'(r_i)/(4 Xi)', sp.simplify(Ti * Si - sp.diff(Dr, r) / (4 * Xi)))
ck.zero('KN-AdS: sum_i 1/(T_i S_i) = 4 Xi sum_i 1/Delta_r\'(r_i) = 0', res_inf(4 * Xi / Dr))
ck.zero('KN-AdS: sum_i 1/T_i = 4 pi sum_i (r_i^2 + a^2)/Delta_r\'(r_i) = 0', res_inf(4 * sp.pi * W / Dr))
Dm = sp.Poly(sp.expand(l ** 2 * Dr), r)                    # monic
prodW = sp.resultant(Dm.as_expr(), W, r)                   # = prod_i (r_i^2 + a^2) for monic Dm
ck.zero('KN-AdS: prod_i (r_i^2 + a^2) = resultant = l^4 Delta_r(ia) Delta_r(-ia) = l^4 (q^4 + 4 m^2 a^2)',
        sp.expand(prodW - l ** 4 * (q ** 4 + 4 * m ** 2 * a ** 2)))
ck.zero('KN-AdS: prod_i S_i = pi^4 l^4 (Q^4 + 4 J^2)', sp.simplify((sp.pi / Xi) ** 4 * prodW - sp.pi ** 4 * l ** 4 * (Qt ** 4 + 4 * Jt ** 2)))
cD = Dm.all_coeffs()
sum_r2 = cD[1] ** 2 - 2 * cD[2]
ck.zero('KN-AdS: sum_i S_i = (pi/Xi)(sum r_i^2 + 4 a^2) = -2 pi l^2', sp.simplify(sp.pi / Xi * (sum_r2 + 4 * a ** 2) + 2 * sp.pi * l ** 2))

# ------------------------------------------------------------------------------------------------- (C) RN-AdS_D
for D in (5, 6):
    PD = r ** (2 * (D - 2)) / l ** 2 + r ** (2 * (D - 3)) - mu * r ** (D - 3) + q ** 2
    # T_i S_i is proportional to P'(r_i) r_i^{-(D-4)}... : 1/(T_i S_i) ~ r_i^{D-4}/P'(r_i), 1/T_i ~ r_i^{2(D-3)}/P'(r_i)
    ck.zero('D = %d RN-AdS: sum_i r_i^{D-4}/P\'(r_i) = 0 (sum of 1/(T_i S_i))' % D, res_inf(r ** (D - 4) / PD))
    ck.zero('D = %d RN-AdS: sum_i r_i^{2(D-3)}/P\'(r_i) = 0 (sum of 1/T_i)' % D, res_inf(r ** (2 * (D - 3)) / PD))
# Algebraic controls: wrong root transport and the rotating frame at infinity.
wrong_root_transport = T * sp.diff(S, r) * (-droot(P, M)) - 1
ck.true('negative control: reversing root displacement fails RN-AdS first law',
        sp.simplify(mod_P(wrong_root_transport, P)) != 0)
Om_rotating = a * Xi / W
wrong_frame = (Ti * (sp.diff(Si, m) + sp.diff(Si, r) * droot(Dr, m))
               + Om_rotating * sp.diff(Jt, m) + Ph * sp.diff(Qt, m) - sp.diff(Mt, m))
ck.true('negative control: rotating-frame Omega with nonrotating-frame mass fails',
        sp.simplify(mod_P(wrong_frame, Dr)) != 0)
ck.zero('quotient-ring control: a regular multiple of P vanishes at every root', mod_P(P / (r ** 2 + 1), P))
for label, expr, polynomial in [('pole at a root', 1 / P, P), ('repeated roots', r, P ** 2)]:
    try:
        mod_P(expr, polynomial)
    except ValueError:
        ck.true('negative control: quotient reduction rejects ' + label, True)
    else:
        ck.true('negative control: quotient reduction rejects ' + label, False)
ck.finish()
