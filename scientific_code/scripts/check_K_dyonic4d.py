r"""
check_K_dyonic4d.py

4D, one scalar with sigma-model metric g(phi), one U(1) with f(phi) and axionic ftilde(phi),
dyonic charges (Q,P), scalar potential V(phi).
    F = E dr^dt + P sin(th) dth^dph ,  G = f *F - ftilde F = Q sin dth^dph + Htil dr^dt ,
    E = (Q + ftilde P)/(f B^2),   Htil = -f P/B^2 - ftilde E  (= d At_dual / dr).
We test whether
    K(r) = (1/2) B [A' dB - B' dA - 2 A dB'] - A B^2 g phi' dphi - At dQ + c * Atil dP
is r-independent on-shell for variations with dP != 0, and find c.
"""
import sympy as sp
from cps import inv_metric, christoffel
from check_K_conservation import ricci

r, t, th, phc = sp.symbols('r t theta varphi')
A = sp.Function('A')(r)
B = sp.Function('B')(r)
ph = sp.Function('phi')(r)
Q, P = sp.symbols('Q P')
dA = sp.Function('dA')(r)
dB = sp.Function('dB')(r)
dph = sp.Function('dphi')(r)
dQ, dP = sp.symbols('dQ dP')
f = sp.Function('f'); ft = sp.Function('ft'); V = sp.Function('V'); gs = sp.Function('g')

x = [t, r, th, phc]
g = sp.diag(-A, 1 / A, B ** 2, B ** 2 * sp.sin(th) ** 2)
ginv = inv_metric(g)
E = (Q + ft(ph) * P) / (f(ph) * B ** 2)
F = sp.zeros(4, 4)
F[1, 0] = E; F[0, 1] = -E
F[2, 3] = P * sp.sin(th); F[3, 2] = -P * sp.sin(th)
Ric = ricci(g, x)
FF = sum(F[m, n] * F[p, q] * ginv[m, p] * ginv[n, q] for m in range(4) for n in range(4) for p in range(4) for q in range(4))
FF = sp.simplify(FF)
eqs = {}
for (m, n) in [(0, 0), (1, 1), (2, 2)]:
    FmFn = sum(F[m, p] * F[n, q] * ginv[p, q] for p in range(4) for q in range(4))
    T = 2 * gs(ph) * sp.diff(ph, x[m]) * sp.diff(ph, x[n]) + 2 * f(ph) * FmFn - g[m, n] * f(ph) * FF / 2 + V(ph) * g[m, n]
    eqs[(m, n)] = sp.simplify((Ric[m, n] - T) * ginv[m, m])
sg = B ** 2
box = sp.diff(sg * gs(ph) * A * sp.diff(ph, r), r) / sg
calL_phi = sp.diff(-f(ph) * FF + 2 * ft(ph) * (-2 * E * P / B ** 2) - 2 * V(ph), ph)  # d/dphi at fixed F (E fixed!)
# careful: d/dphi above also differentiates E through f, ft; we need d/dphi at fixed F, so build with a frozen E
Es = sp.Symbol('Es')
FFs = FF.subs(E, Es)
calL_phi = (sp.diff(-f(ph) * FFs + 2 * ft(ph) * (-2 * Es * P / B ** 2) - 2 * V(ph), ph)).subs(Es, E)
scal = 4 * box - 2 * sp.diff(gs(ph), ph) * A * sp.diff(ph, r) ** 2 + calL_phi
eqs['phi'] = sp.simplify(scal)


def linearise(expr, e=sp.Symbol('e')):
    rep = {A: A + e * dA, B: B + e * dB, ph: ph + e * dph, Q: Q + e * dQ, P: P + e * dP}
    return sp.diff(expr.subs(rep, simultaneous=True), e).subs(e, 0)


A2, B2, P2 = sp.symbols('A2 B2 P2')
sub2 = {sp.diff(A, r, 2): A2, sp.diff(B, r, 2): B2, sp.diff(ph, r, 2): P2}
sol = sp.solve([eqs[(0, 0)].subs(sub2), eqs[(2, 2)].subs(sub2), eqs['phi'].subs(sub2)], [A2, B2, P2], dict=True)[0]
constraint = sp.simplify((eqs[(1, 1)] - eqs[(0, 0)]).subs(sub2).subs(sol))
dA2, dB2, dP2 = sp.symbols('dA2 dB2 dP2')
subd2 = {sp.diff(dA, r, 2): dA2, sp.diff(dB, r, 2): dB2, sp.diff(dph, r, 2): dP2}
linsys = [linearise(eqs[k]).subs(subd2).subs(sub2).subs(sol) for k in [(0, 0), (2, 2), 'phi']]
dsol = sp.solve(linsys, [dA2, dB2, dP2], dict=True)[0]
lc = linearise(eqs[(1, 1)] - eqs[(0, 0)]).subs(subd2).subs(sub2).subs(sol).subs(dsol)

if __name__ == '__main__':
    import checklib as ck
    c = sp.Symbol('c')
    At = sp.Function('At')(r)
    Atil = sp.Function('Atil')(r)
    Htil = -f(ph) * P / B ** 2 - ft(ph) * E
    K = sp.Rational(1, 2) * B * (sp.diff(A, r) * dB - sp.diff(B, r) * dA - 2 * A * sp.diff(dB, r)) \
        - A * B ** 2 * gs(ph) * sp.diff(ph, r) * dph - At * dQ + c * Atil * dP
    Kp = sp.diff(K, r).subs({sp.diff(At, r): E, sp.diff(Atil, r): Htil})
    Kp = Kp.subs(sub2).subs(sol).subs(subd2).subs(dsol)
    Ap, dAp = sp.symbols('Ap dAp')
    # eliminate A' with the background constraint and dA' with the linearised constraint (both are linear in them)
    Apval = sp.solve(constraint.subs(sp.diff(A, r), Ap), Ap)[0]
    lc1 = lc.subs(sp.diff(dA, r), dAp)
    dApval = sp.solve(lc1, dAp)[0]
    Kp = Kp.subs(sp.diff(dA, r), dApval)
    Kp = Kp.subs(sp.diff(A, r), Apval)
    Kp = sp.simplify(sp.expand(Kp))
    print('dK/dr on-shell =', Kp)
    print('coefficient of dP:', sp.simplify(sp.diff(Kp, dP)))
    print('solve for c:', sp.solve(sp.diff(Kp, dP), c))
    print('remaining (c=1):', sp.simplify(Kp.subs(c, 1)))
    csol = sp.solve(sp.diff(Kp, dP), c)
    ck.true('the magnetic completion coefficient is c = 1', csol == [1], 'solution %s' % csol)
    ck.zero('dK/dr = 0 on shell with c = 1 (dyonic, axionic ftilde)', Kp.subs(c, 1))
    ck.true('negative control: without the completion (c = 0) dK/dr is not zero', sp.simplify(Kp.subs(c, 0)) != 0)
    ck.finish()
