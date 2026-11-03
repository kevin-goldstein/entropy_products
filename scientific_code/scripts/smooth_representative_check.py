r"""
smooth_representative_check.py -- the smoothing vector field of Section 2.3 for a rotating bifurcation surface.

Local model of a non-extremal rotating Killing horizon near its bifurcation surface s = 0:
    g = -kappa^2 s^2 dt^2 + ds^2 + R^2 (dphi - Omega dt)^2 ,      xi = d_t + Omega d_phi .
Smooth (Kruskal-type) coordinates:  U = -s e^{-kappa t},  V = s e^{kappa t},  phit = phi - Omega t, in which
    g = -dU dV + R^2 dphit^2  (exactly), so a perturbation is smooth at B = {U = V = 0} iff its components in
(U, V, phit) are smooth there.  The Boyer-Lindquist-type variations delta kappa, delta Omega at fixed (t, s, phi) give
    delta g = -2 kappa delta kappa s^2 dt^2 - 2 R^2 delta Omega dt (dphi - Omega dt) + 2 R delta R (dphi - Omega dt)^2 ,
whose dt-terms are singular at B.  We verify that
    X = -(delta kappa/kappa) t xi + delta Omega t d_phi         (rescaling along the horizon generator, PLUS sign)
makes delta g + L_X g smooth (polynomial in U, V), while the negative-control vector
-(delta kappa/kappa) t d_t - delta Omega t d_phi, leaves (indeed doubles) the singular delta-Omega term.
The inner bifurcation surface is checked in the same way with the region-II local model
    g = +kappa^2 s^2 dt^2 - ds^2 + R^2 (dphi - Omega dt)^2,   U = s e^{-kappa t},  V = s e^{kappa t}   (g = -dU dV + ...),
where the radial direction is timelike.
"""
import sympy as sp
import checklib as ck

t, s, ph = sp.symbols('t s phi', real=True)
U, V, pt = sp.symbols('U V phit', real=True)
kap, Om, R = sp.symbols('kappa Omega R', positive=True)
dk, dO, dR = sp.symbols('delta_kappa delta_Omega delta_R', real=True)
X0 = [t, s, ph]


def run_horizon(sig, tag):
    """sig = +1: outer horizon (region I side, s spacelike); sig = -1: inner horizon seen from region II (s timelike)."""
    def metric(k, O, Rr):
        return sp.Matrix([[-sig * k ** 2 * s ** 2 + Rr ** 2 * O ** 2, 0, -Rr ** 2 * O], [0, sig, 0],
                          [-Rr ** 2 * O, 0, Rr ** 2]])
    g = metric(kap, Om, R)
    lam = sp.Symbol('lam')
    dg = sp.diff(metric(kap + lam * dk, Om + lam * dO, R + lam * dR), lam).subs(lam, 0)

    def lie(Xv):
        """(L_X g)_{mn} = X^r d_r g_{mn} + g_{rn} d_m X^r + g_{mr} d_n X^r"""
        L = sp.zeros(3, 3)
        for m in range(3):
            for n in range(3):
                L[m, n] = sum(Xv[r_] * sp.diff(g[m, n], X0[r_]) + g[r_, n] * sp.diff(Xv[r_], X0[m])
                              + g[m, r_] * sp.diff(Xv[r_], X0[n]) for r_ in range(3))
        return L
    # Jacobian d(t, s, phi)/d(U, V, phit): outer U < 0 < V, s^2 = -U V;  inner U, V > 0, s^2 = U V
    t_UV = sp.log(-sig * V / U) / (2 * kap)
    s_UV = sp.sqrt(-sig * U * V)
    ph_UV = pt + Om * t_UV
    J = sp.Matrix([[sp.diff(f, y) for y in (U, V, pt)] for f in (t_UV, s_UV, ph_UV)])
    sub = {t: t_UV, s: s_UV, ph: ph_UV}

    def kruskal(h):
        return (J.T * h.subs(sub) * J).applyfunc(lambda e: sp.simplify(sp.expand(e)))

    gk = kruskal(g)
    ck.zero('%s: background g = -dU dV + R^2 dphit^2 in Kruskal coordinates' % tag,
            sum((gk - sp.Matrix([[0, -sp.Rational(1, 2), 0], [-sp.Rational(1, 2), 0, 0], [0, 0, R ** 2]])).applyfunc(sp.Abs)))
    xi = [1, 0, Om]
    q = dk / kap
    X_smooth = [-q * t * xi[0], 0, -q * t * xi[2] + dO * t]
    X_negative = [-q * t, 0, -dO * t]
    for lab, Xv in (('no smoothing vector', [0, 0, 0]), ('negative control: -(dk/k) t d_t - dOmega t d_phi', X_negative),
                    ('smooth: -(dk/k) t xi + dOmega t d_phi', X_smooth)):
        hk = kruskal(dg + lie(Xv))
        smooth = all(sp.simplify(e).is_polynomial(U, V) for e in hk)
        print('%s, %-50s  components in (U, V, phit):' % (tag, lab))
        for (i, j) in ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)):
            print('      h_%s%s = %s' % ('UVp'[i], 'UVp'[j], sp.simplify(hk[i, j])))
        if lab.startswith('smooth'):
            ck.true('%s, smooth vector: delta g + L_X g is polynomial in (U, V), i.e. smooth at the bifurcation surface'
                    % tag, smooth)
            ck.zero('%s, smooth vector: the delta-Omega and delta-kappa parts cancel (only 2 R dR dphit^2 is left)' % tag,
                    sum((hk - sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 2 * R * dR]])).applyfunc(sp.Abs)))
        else:
            ck.true('%s, %s: delta g + L_X g is NOT smooth at the bifurcation surface' % (tag, lab), not smooth)
    h_bare = kruskal(dg.subs(dk, 0))
    h_negative = kruskal((dg + lie(X_negative)).subs(dk, 0))
    ck.zero('%s: with delta kappa = 0 the negative-control vector doubles the singular delta-Omega terms' % tag,
            sp.simplify(h_negative[0, 2] - 2 * h_bare[0, 2]) + sp.simplify(h_negative[1, 2] - 2 * h_bare[1, 2]))
    comm = [sum(xi[r_] * sp.diff(X_smooth[m], X0[r_]) - X_smooth[r_] * sp.diff(xi[m], X0[r_]) for r_ in range(3)) for m in range(3)]
    ck.zero('%s: [xi, X] = -(dk/k) xi + dOmega d_phi, a combination of the Killing vectors d_t, d_phi' % tag,
            sum(sp.Abs(sp.simplify(comm[m] - (-q * xi[m] + (dO if m == 2 else 0)))) for m in range(3)))


run_horizon(+1, 'outer B+')
run_horizon(-1, 'inner B-')
ck.finish()
