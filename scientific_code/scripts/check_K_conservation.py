r"""
check_K_conservation.py

Verify that the radial Noether function
    K(r) = (D-2) Omega_{D-2}/(16 pi) B^{D-3} [ A' dB - B' dA - 2 A dB' ]
           - Omega_{D-2}/(4 pi) A B^{D-2} g(phi) phi' dphi
           - A_t dQ
is r-independent for every solution (A,B,phi,E) of the static field equations and every
solution (dA,dB,dphi,dQ) of the linearised equations, in D = 4,5,6 dimensions, for the theory
    L = (1/16pi)[R - 2 g(phi)(dphi)^2 - f(phi) F^2 - 2 V(phi)]
with a purely electric field  F = E dr ^ dt,  A_t' = E,  E = 4 pi Q /(Omega f B^{D-2}).
"""
import sympy as sp
from cps import inv_metric, christoffel, sphere_metric, sphere_volume

r, t = sp.symbols('r t')
A = sp.Function('A')(r)
B = sp.Function('B')(r)
ph = sp.Function('phi')(r)
Q = sp.Symbol('Q')
dA = sp.Function('dA')(r)
dB = sp.Function('dB')(r)
dph = sp.Function('dphi')(r)
dQ = sp.Symbol('dQ')
f = sp.Function('f')
V = sp.Function('V')
gs = sp.Function('g')   # sigma-model metric g(phi) for one scalar


def ricci(g, x):
    D = len(x)
    ginv = inv_metric(g)
    Gam = christoffel(g, ginv, x)
    Ric = sp.zeros(D, D)
    for b in range(D):
        for c in range(b, D):
            s = 0
            for a in range(D):
                s += sp.diff(Gam[a][b][c], x[a]) - sp.diff(Gam[a][b][a], x[c])
                for d in range(D):
                    s += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
            Ric[b, c] = sp.simplify(s)
            Ric[c, b] = Ric[b, c]
    return Ric


def field_equations(D):
    angles = sp.symbols('th1:%d' % (D - 1))
    x = [t, r] + list(angles)
    sph = sphere_metric(D - 2, angles)
    g = sp.diag(-A, 1 / A, *[B ** 2 * s for s in sph])
    ginv = inv_metric(g)
    Om = sphere_volume(D - 2)
    E = 4 * sp.pi * Q / (Om * f(ph) * B ** (D - 2))
    Ric = ricci(g, x)
    # F_{rt} = E ; F_{tr} = -E
    F = sp.zeros(D, D)
    F[1, 0] = E
    F[0, 1] = -E
    FF = sum(F[m, n] * F[p, q] * ginv[m, p] * ginv[n, q] for m in range(D) for n in range(D) for p in range(D) for q in range(D))
    eqs = {}
    for (m, n) in [(0, 0), (1, 1), (2, 2)]:
        FmFn = sum(F[m, p] * F[n, q] * ginv[p, q] for p in range(D) for q in range(D))
        dphm = sp.diff(ph, x[m])
        dphn = sp.diff(ph, x[n])
        T = 2 * gs(ph) * dphm * dphn + 2 * f(ph) * FmFn - g[m, n] * f(ph) * FF / (D - 2) + 2 * V(ph) * g[m, n] / (D - 2)
        eqs[(m, n)] = sp.simplify((Ric[m, n] - T) * ginv[m, m])   # mixed components E^m_m
    # scalar equation: 4 nabla(g nabla phi) - 2 g' (dphi)^2 - f' F^2 - 2 V' = 0
    sg = sp.sqrt(A * (1 / A)) * B ** (D - 2)
    box = sp.diff(sg * gs(ph) * ginv[1, 1] * sp.diff(ph, r), r) / sg
    scal = 4 * box - 2 * sp.diff(gs(ph), ph) * ginv[1, 1] * sp.diff(ph, r) ** 2 - sp.diff(f(ph), ph) * FF - 2 * sp.diff(V(ph), ph)
    eqs['phi'] = sp.simplify(scal)
    return eqs, E


def linearise(expr, e=sp.Symbol('e')):
    rep = {A: A + e * dA, B: B + e * dB, ph: ph + e * dph, Q: Q + e * dQ}
    return sp.diff(expr.subs(rep, simultaneous=True), e).subs(e, 0)


def check(D, gauge_coeff=1):
    eqs, E = field_equations(D)
    Om = sphere_volume(D - 2)
    # second-derivative solutions
    A2, B2, P2 = sp.symbols('A2 B2 P2')
    sub2 = {sp.diff(A, r, 2): A2, sp.diff(B, r, 2): B2, sp.diff(ph, r, 2): P2}
    E_tt = eqs[(0, 0)].subs(sub2)
    E_rr = eqs[(1, 1)].subs(sub2)
    E_th = eqs[(2, 2)].subs(sub2)
    E_ph = eqs['phi'].subs(sub2)
    sol = sp.solve([E_tt, E_th, E_ph], [A2, B2, P2], dict=True)[0]
    # constraint = E_rr - E_tt (first order)
    constraint = sp.simplify((E_rr - E_tt).subs(sol))
    # K(r)
    At = sp.Function('At')(r)
    K = (D - 2) * Om / (16 * sp.pi) * B ** (D - 3) * (sp.diff(A, r) * dB - sp.diff(B, r) * dA - 2 * A * sp.diff(dB, r)) \
        - Om / (4 * sp.pi) * A * B ** (D - 2) * gs(ph) * sp.diff(ph, r) * dph - gauge_coeff * At * dQ
    Kp = sp.diff(K, r).subs(sp.diff(At, r), E)
    # linearised second derivatives
    dA2, dB2, dP2 = sp.symbols('dA2 dB2 dP2')
    lin = {}
    for key, ex in [('tt', eqs[(0, 0)]), ('th', eqs[(2, 2)]), ('phi', eqs['phi'])]:
        lin[key] = linearise(ex)
    subd2 = {sp.diff(dA, r, 2): dA2, sp.diff(dB, r, 2): dB2, sp.diff(dph, r, 2): dP2}
    linsys = [ (lin[k].subs(subd2)).subs(sub2).subs(sol) for k in ('tt', 'th', 'phi')]
    # background second derivatives inside linearised eqs
    dsol = sp.solve(linsys, [dA2, dB2, dP2], dict=True)[0]
    Kp2 = Kp.subs(sub2).subs(sol).subs(subd2).subs(dsol)
    Kp2 = sp.simplify(Kp2)
    # reduce modulo the constraint and its linearisation
    lin_constraint = sp.simplify(linearise(eqs[(1, 1)] - eqs[(0, 0)]).subs(subd2).subs(sub2).subs(sol).subs(dsol))
    return sp.simplify(Kp2), sp.simplify(constraint), lin_constraint


def on_shell_dK(D, gauge_coeff=1):
    Ap, dAp = sp.symbols('Ap dAp')
    Kp, c, lc = check(D, gauge_coeff)
    Apval = sp.solve(c.subs(sp.diff(A, r), Ap), Ap)[0]
    dApval = sp.solve(lc.subs(sp.diff(dA, r), dAp), dAp)[0]
    Kp2 = Kp.subs(sp.diff(dA, r), dApval).subs(sp.diff(A, r), Apval)
    return sp.simplify(sp.expand(Kp2))


if __name__ == '__main__':
    import checklib as ck
    for D in (4, 5, 6):
        Kp2 = on_shell_dK(D)
        print('D =', D, ':  dK/dr on-shell =', Kp2)
        ck.zero('D=%d: dK/dr = 0 on shell' % D, Kp2)
    ctrl = on_shell_dK(4, gauge_coeff=0)
    ck.true('negative control: without the -A_t dQ term dK/dr is not zero (D=4)', sp.simplify(ctrl) != 0,
            'dK/dr = %s' % sp.factor(ctrl))
    ck.finish()
