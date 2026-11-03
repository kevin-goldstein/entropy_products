r"""
btz_check.py -- BTZ (G = 1, L = (R + 2/l^2)/16pi): covariant-phase-space charges and the Cardy form of S_pm.
    N^2 = -8M + r^2/l^2 + 16 J^2/r^2,  N^phi = -4J/r^2,  ds^2 = -N^2 dt^2 + dr^2/N^2 + r^2 (dphi + N^phi dt)^2
Checks: K_t = dM and K_phi = -dJ at all r (outside, between and inside the horizons); first laws at r_pm;
S_pm = 2 pi ( sqrt(c L0/6) +- sqrt(c L0bar/6) ) with c = 3l/2, L0 = (Ml+J)/2, L0bar = (Ml-J)/2;
S+S- = 4 pi^2 (c/6)(L0 - L0bar) = pi^2 l J = S_ext(J)^2.
"""
import sympy as sp
import checklib as ck
from cps import k_grav_tr

t, r, ph = sp.symbols('t r phi', real=True)
M, J, l = sp.symbols('M J l', positive=True)
eps = sp.Symbol('epsilon_var')
N2 = -8 * M + r ** 2 / l ** 2 + 16 * J ** 2 / r ** 2
Nph = -4 * J / r ** 2
g = sp.Matrix([[-N2 + r ** 2 * Nph ** 2, 0, r ** 2 * Nph], [0, 1 / N2, 0], [r ** 2 * Nph, 0, r ** 2]])
x = [t, r, ph]


def K(var, xi):
    h = g.diff(var)
    k16 = k_grav_tr(g, h, x, xi, eps=eps, sqrtg=lambda ge, e: sp.sqrt(-ge.det()))
    return sp.simplify(2 * sp.pi * k16 / (16 * sp.pi))


rho = sp.Symbol('rho', positive=True)
expect = {(M, 't'): 1, (M, 'phi'): 0, (J, 't'): 0, (J, 'phi'): -1}
for var in (M, J):
    for name, xi in (('t', [1, 0, 0]), ('phi', [0, 0, 1])):
        Kv = sp.simplify(K(var, xi))
        print('d/d%s  K_%s(r) =' % (var, name), Kv)
        ck.zero('BTZ: d/d%s K_%s = %s for all r > 0' % (var, name, expect[(var, name)]), Kv.subs(r, rho) - expect[(var, name)])

rp2 = 4 * M * l ** 2 * (1 + sp.sqrt(1 - J ** 2 / (M ** 2 * l ** 2)))
rm2 = 4 * M * l ** 2 * (1 - sp.sqrt(1 - J ** 2 / (M ** 2 * l ** 2)))
rp, rm = sp.sqrt(rp2), sp.sqrt(rm2)
S = lambda rh: sp.pi * rh / 2
T = lambda rh, ro: (rh ** 2 - ro ** 2) / (2 * sp.pi * l ** 2 * rh)
Om = lambda rh, ro: ro / (l * rh)
c = sp.Rational(3, 2) * l
L0, L0b = (M * l + J) / 2, (M * l - J) / 2
num = {M: sp.Rational(7, 5), J: sp.Rational(1, 2), l: sp.Rational(3, 2)}
for lab, rh, ro in (('+', rp, rm), ('-', rm, rp)):
    for var in (M, J):
        lhs = (1 if var == M else 0) - Om(rh, ro) * (1 if var == J else 0)
        rhs = T(rh, ro) * sp.diff(S(rh), var)
        print('first law at r%s for d/d%s: residual =' % (lab, var), sp.N((lhs - rhs).subs(num), 15))
        ck.small('BTZ: first law at r%s for d/d%s' % (lab, var), sp.N((lhs - rhs).subs(num), 60), 1e-50)
cp = sp.N((S(rp) - 2 * sp.pi * (sp.sqrt(c * L0 / 6) + sp.sqrt(c * L0b / 6))).subs(num), 60)
cm = sp.N((S(rm) - 2 * sp.pi * (sp.sqrt(c * L0 / 6) - sp.sqrt(c * L0b / 6))).subs(num), 60)
pr = sp.N((S(rp) * S(rm) - sp.pi ** 2 * l * J).subs(num), 60)
ts = sp.N((T(rp, rm) * S(rp) + T(rm, rp) * S(rm)).subs(num), 60)
print('Cardy S+ :', sp.N(cp, 15))
print('Cardy S- :', sp.N(cm, 15))
print('S+S- - pi^2 l J :', sp.N(pr, 15))
print('T+S+ + T-S- :', sp.N(ts, 15), '  (signed T_- = -(r+^2-r-^2)/(2 pi l^2 r-))')
ck.small('BTZ: Cardy form of S+', cp, 1e-50)
ck.small('BTZ: Cardy form of S-', cm, 1e-50)
ck.small('BTZ: S+S- = pi^2 l J', pr, 1e-50)
ck.small('BTZ: T+S+ + T-S- = 0', ts, 1e-50)
ck.finish()
