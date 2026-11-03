r"""
subtracted_rn.py -- metric-level check that the subtracted geometry of the 4-equal-charge (RN) STU black hole
preserves both horizons' (S, T), the 3D base metric, and that the GJN local rescaling (1512.08685) preserves
T_pm S_pm and S+S- but not S_pm separately.
    ds^2 = -Delta^{-1/2} X dt^2 + Delta^{1/2}(dr^2/X + dOmega^2),  X = r^2 - 2 m r,
    RN:          Delta   = (r + 2 m s^2)^4                     (s = sinh delta, c = cosh delta)
    subtracted:  Delta_0 = (2m)^3 r (c^8 - s^8) + (2m)^4 s^8   (Cvetic-Larsen; Cvetic-Gibbons)
"""
import sympy as sp
import checklib as ck
r, m, dl = sp.symbols('r m delta', positive=True)
s, c = sp.sinh(dl), sp.cosh(dl)
X = r ** 2 - 2 * m * r
D_rn = (r + 2 * m * s ** 2) ** 4
D_sub = (2 * m) ** 3 * r * (c ** 8 - s ** 8) + (2 * m) ** 4 * s ** 8
def horizon_data(Dl):
    # g_tt = -X/sqrt(Dl);  area = 4 pi sqrt(Dl);  T = X'/(4 pi sqrt(Dl))  (signed)
    out = {}
    for lab, rh in (('+', 2 * m), ('-', 0)):
        S = sp.pi * sp.sqrt(Dl.subs(r, rh))
        T = sp.diff(X, r).subs(r, rh) / (4 * sp.pi * sp.sqrt(Dl.subs(r, rh)))
        out[lab] = (sp.simplify(S), sp.simplify(T))
    return out
H1, H2 = horizon_data(D_rn), horizon_data(D_sub)
for lab in '+-':
    print('horizon %s: S_RN - S_sub =' % lab, sp.simplify(H1[lab][0] - H2[lab][0]), ';  T_RN - T_sub =', sp.simplify(H1[lab][1] - H2[lab][1]))
    ck.zero('subtraction preserves S at the %s horizon' % lab, (H1[lab][0] - H2[lab][0]).rewrite(sp.exp))
    ck.zero('subtraction preserves T at the %s horizon' % lab, (H1[lab][1] - H2[lab][1]).rewrite(sp.exp))
print('3D base metric gamma = dr^2 + X dOmega^2 for both (g_tt g_rr = -1/ ... ):', 'X =', X)
# GJN local rescaling on RN in areal gauge: a^2 = (r-r+)(r-r-)/r^2, b = r  ->  lambda = r/sqrt(r+ r-)
rp, rm = sp.symbols('r_p r_m', positive=True)
a2 = (r - rp) * (r - rm) / r ** 2; b = r
lam = r / sp.sqrt(rp * rm)
a2n, bn = sp.simplify(lam ** 2 * a2), sp.simplify(b / lam)
for lab, rh in (('+', rp), ('-', rm)):
    T0, S0 = sp.diff(a2, r).subs(r, rh) / (4 * sp.pi), sp.pi * b.subs(r, rh) ** 2
    T1, S1 = sp.diff(a2n, r).subs(r, rh) / (4 * sp.pi), sp.pi * bn.subs(r, rh) ** 2
    print('rescaling, horizon %s: T S before/after:' % lab, sp.simplify(T0 * S0), '/', sp.simplify(T1 * S1), ';  S before/after:', S0, '/', S1)
    ck.zero('GJN rescaling preserves T S at the %s horizon' % lab, T0 * S0 - T1 * S1)
    ck.true('GJN rescaling changes S at the %s horizon' % lab, sp.simplify(S0 - S1) != 0)
    if lab == '+':
        Sp0, Sp1 = S0, S1
    else:
        ck.zero('GJN rescaling preserves S+S-', Sp0 * S0 - Sp1 * S1)

# Lift of the AdS2 x S2 black hole with constant field strength: the 3D part of the 5D metric is
#   -(r^2 - Dl^2)/L^2 dt^2 + L^2 dr^2/(r^2 - Dl^2) + R^2 (dy + k r dt)^2 ,
# which is locally AdS3 (Ric = -(2/ell^2) g) iff |k| = 1/(L R), with ell = 2L, for every Dl (k is taken positive below;
# k -> -k is y -> -y).
# With the standard untwisted identification y ~ y + 2 pi, the fibre generator lies in one SL(2,R) factor.
# The usual extension is the spacelike self-dual quotient (Coussaert-Henneaux), rather than generic BTZ.
# This local curvature calculation does not specify a global extension or validate a microscopic Cardy regime.
t_, y_ = sp.symbols('t y', real=True)
L_, R_, k_, Dl_, ell_ = sp.symbols('L R k Delta ell', positive=True)
Xc = [t_, r, y_]


def ricci3(metric):
    """Ricci tensor from the metric in (t, r, y), with no field equations imposed."""
    inverse = metric.inv()
    connection = [[[sp.simplify(sum(inverse[a, d] * (
        sp.diff(metric[d, b], Xc[c]) + sp.diff(metric[d, c], Xc[b])
        - sp.diff(metric[b, c], Xc[d])) for d in range(3)) / 2)
        for c in range(3)] for b in range(3)] for a in range(3)]
    return sp.Matrix(3, 3, lambda a, b: sp.simplify(sum(
        sp.diff(connection[c][a][b], Xc[c]) - sp.diff(connection[c][a][c], Xc[b])
        + sum(connection[c][c][d] * connection[d][a][b]
              - connection[c][b][d] * connection[d][a][c] for d in range(3))
        for c in range(3))))


f_ = (r ** 2 - Dl_ ** 2) / L_ ** 2
g3 = sp.Matrix([[-f_ + R_ ** 2 * k_ ** 2 * r ** 2, 0, R_ ** 2 * k_ * r], [0, 1 / f_, 0], [R_ ** 2 * k_ * r, 0, R_ ** 2]])
g3i = g3.inv()
Ric3 = ricci3(g3)
sol3 = sp.solve([sp.simplify(Ric3[a_, b_] + 2 / ell_ ** 2 * g3[a_, b_]) for a_ in range(3) for b_ in range(3)],
                [k_, ell_], dict=True)
print('lift of the AdS2 x S2 black hole: locally AdS3 iff', sol3, '(independent of Delta)')
ck.true('lift: Ric = -(2/ell^2) g has the unique solution k = 1/(L R), ell = 2 L for every Delta',
        len(sol3) == 1 and sp.simplify(sol3[0][k_] - 1 / (L_ * R_)) == 0 and sp.simplify(sol3[0][ell_] - 2 * L_) == 0)
ck.zero('constant-fibre lift: scalar curvature is -6/(2 L)^2',
        sp.trace(g3i * Ric3).subs(k_, 1 / (L_ * R_)) + sp.Rational(3, 2) / L_**2)

# Negative control: the THREE-DIMENSIONAL part of the metric printed in
# Goldstein-Jejjala-Nampuri, arXiv:1410.3478, Eq. (5.3), https://arxiv.org/pdf/1410.3478
#   ds3^2 = -f dt^2 + dr^2/f + r^2 (dy + r_p r_m/(L r) dt)^2,
#   f = (r-r_p)(r-r_m)/L^2.
# This is not the constant-fibre metric checked above. Its nonconstant Ricci scalar
# rules out a local BTZ interpretation; no global-identification assumption is involved.
f_printed = (r - rp) * (r - rm) / L_**2
A_printed = rp * rm / (L_ * r)
g_printed = sp.Matrix([
    [-f_printed + r**2 * A_printed**2, 0, r**2 * A_printed],
    [0, 1 / f_printed, 0],
    [r**2 * A_printed, 0, r**2]])
ricci_printed = ricci3(g_printed)
scalar_printed = sp.factor(sp.trace(g_printed.inv() * ricci_printed))
scalar_expected = (-6 / L_**2 + 2 * (rp + rm) / (L_**2 * r)
                   + (rp * rm)**2 / (2 * L_**2 * r**2))
print('GJN2014 Eq. (5.3), printed 3D metric: Ricci scalar =', scalar_printed)
ck.zero('printed GJN2014 metric: exact nonconstant Ricci scalar', scalar_printed - scalar_expected)
ck.true('printed GJN2014 metric: d(Ricci scalar)/dr < 0 for positive r, r_p, r_m, L',
        sp.factor(sp.diff(scalar_printed, r)).is_negative)
tracefree_yy = sp.factor(ricci_printed[2, 2] - scalar_printed * g_printed[2, 2] / 3)
ck.zero('printed GJN2014 metric: exact trace-free yy Ricci component',
        tracefree_yy - ((rp + rm) * r - 2 * (rp * rm)**2) / (3 * L_**2))
# This exterior point also obeys the RN relation L^2 = r_p r_m.
ck.large('negative control: printed GJN2014 metric is not Einstein even with L^2 = r_p r_m',
         tracefree_yy.subs({rp: 2, rm: sp.Rational(1, 2), L_: 1, r: 3}), 1)
ck.finish()
