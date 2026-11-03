r"""
reduced_equations_check.py -- verify the reduced static equations of sec. 3.1 of the paper against the covariant
field equations, for D = 4, one scalar with arbitrary g(phi), f(phi), ftilde(phi), V(phi), dyonic charges (Q,P):
  (red1)  (a^2 b^2)'' = 2 - 4 b^2 V,          b''/b = - g phi'^2
  (red2)  (a^2 b^2 g phi')' - (1/2) a^2 b^2 g' phi'^2 = Veff'/(2 b^2) + (b^2/2) V'
  (red3)  (1/2)(b^2 (a^2)')' = Veff/b^2 - b^2 V ,    -(1/2)(a^2 (b^2)')' = Veff/b^2 + b^2 V - 1
  (red4)  -1 + a^2 b'^2 + (1/2)(a^2)'(b^2)' = -Veff/b^2 - b^2 V + a^2 b^2 g phi'^2
with Veff = f^{-1}(Q + ft P)^2 + f P^2 and Veff' = d/dphi at fixed (Q,P).
Each reduced equation is checked to be a linear combination of the mixed components E^t_t, E^r_r, E^th_th and the
scalar equation (all computed from first principles in check_K_dyonic4d.py).
"""
import sympy as sp
import check_K_dyonic4d as dy

A, B, ph, r = dy.A, dy.B, dy.ph, dy.r
Q, P = dy.Q, dy.P
f, ft, V, gs = dy.f, dy.ft, dy.V, dy.gs
Ett, Err, Eth, Esc = dy.eqs[(0, 0)], dy.eqs[(1, 1)], dy.eqs[(2, 2)], dy.eqs['phi']
Veff = (Q + ft(ph) * P) ** 2 / f(ph) + f(ph) * P ** 2
x = sp.Symbol('x')
Veffp = sp.diff(((Q + ft(x) * P) ** 2 / f(x) + f(x) * P ** 2), x).subs(x, ph)
Vp = sp.diff(V(x), x).subs(x, ph)
gp = sp.diff(gs(x), x).subs(x, ph)
d = lambda X, n=1: sp.diff(X, r, n)
reduced = {
    'red1a': d(A * B ** 2, 2) - 2 + 4 * B ** 2 * V(ph),
    'red1b': d(B, 2) / B + gs(ph) * d(ph) ** 2,
    'red2': d(A * B ** 2 * gs(ph) * d(ph)) - A * B ** 2 * gp * d(ph) ** 2 / 2 - Veffp / (2 * B ** 2) - B ** 2 * Vp / 2,
    'red3a': d(B ** 2 * d(A)) / 2 - Veff / B ** 2 + B ** 2 * V(ph),
    'red3b': -d(A * d(B ** 2)) / 2 - Veff / B ** 2 - B ** 2 * V(ph) + 1,
    'red4': -1 + A * d(B) ** 2 + d(A) * d(B ** 2) / 2 + Veff / B ** 2 + B ** 2 * V(ph) - A * B ** 2 * gs(ph) * d(ph) ** 2,
}
import checklib as ck
cs = sp.symbols('c1:5')
for name, expr in reduced.items():
    comb = expr - (cs[0] * Ett + cs[1] * Err + cs[2] * Eth + cs[3] * Esc)
    num = sp.numer(sp.together(sp.expand(comb)))
    # treat derivatives of A, B, phi up to 2nd order and the coupling functions as independent symbols
    reps = {}
    syms = []
    for k, obj in enumerate([d(A, 2), d(B, 2), d(ph, 2), d(A), d(B), d(ph), A, B, ph]):
        s_ = sp.Symbol('z%d' % k); reps[obj] = s_; syms.append(s_)
    num = num.subs(reps)
    sol = sp.solve(sp.Poly(sp.expand(num), *syms[:6]).coeffs(), cs, dict=True)
    ok = bool(sol) and sp.simplify(num.subs(sol[0])) == 0
    print('%-5s is a combination of the field equations: %s   coefficients: %s' % (name, ok, sol[0] if sol else None))
    ck.true('%s is a linear combination of the covariant field equations' % name, ok)
ck.finish()
