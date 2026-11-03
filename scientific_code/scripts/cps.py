r"""
cps.py -- minimal covariant-phase-space toolkit (SymPy) used throughout the paper.

Conventions (G = 1 in every dimension D):
    L = (1/16 pi) [ R - 2 g_ij(phi) dphi^i.dphi^j - f_ab(phi) F^a.F^b - 2 V(phi) ] eps
      (+ (1/16 pi) 2 ftilde_ab F^a ^ F^b  in D = 4)
    Theta_grav^mu = (1/16 pi) (nabla_nu h^{mu nu} - nabla^mu h),   Theta = Theta^mu i_mu eps
    Q_grav^{mu nu} = -(1/16 pi) (nabla^mu xi^nu - nabla^nu xi^mu),  Q = (1/2) Q^{mu nu} i_nu i_mu eps
    Theta_phi^mu  = -(1/4 pi) g_ij delta phi^i nabla^mu phi^j
    Theta_A       = -(1/4 pi) delta A^a ^ G_a ,   Q_A = -(1/4 pi) (xi.A^a) G_a ,
    G_a = f_ab *F^b - ftilde_ab F^b      (Maxwell: dG_a = 0)
    k_xi = delta Q_xi - xi.Theta  <->  k^{mu nu} sqrt(-g) = delta(sqrt(-g) Q^{mu nu}) - sqrt(-g)(Theta^mu xi^nu - Theta^nu xi^mu)

The surface integral over the (D-2)-sphere at fixed (t,r) is  K = \oint k^{tr} sqrt(-g) d^{D-2}theta,
up to an overall orientation sign that is fixed once and for all by demanding K(r->infinity) = +delta M.
"""
import sympy as sp


def inv_metric(g):
    # works for any (small) metric; fast path for diagonal ones
    if g.is_diagonal():
        return sp.diag(*[1 / g[i, i] for i in range(g.shape[0])])
    return sp.simplify(g.inv())


def christoffel(g, ginv, x):
    D = len(x)
    dg = [[[sp.diff(g[a, b], x[c]) for c in range(D)] for b in range(D)] for a in range(D)]
    Gam = [[[0] * D for _ in range(D)] for _ in range(D)]
    for a in range(D):
        for b in range(D):
            for c in range(b, D):
                s = 0
                for d in range(D):
                    if ginv[a, d] != 0:
                        s += ginv[a, d] * (dg[d][b][c] + dg[d][c][b] - dg[b][c][d])
                s = s / 2
                Gam[a][b][c] = s
                Gam[a][c][b] = s
    return Gam


def theta_grav(g, ginv, Gam, h, x):
    """Theta^mu (vector) for GR, without the 1/16pi."""
    D = len(x)
    hup = ginv * h * ginv
    htr = sum(ginv[a, b] * h[a, b] for a in range(D) for b in range(D))
    out = []
    for mu in range(D):
        s = 0
        for nu in range(D):
            s += sp.diff(hup[mu, nu], x[nu])
            for lam in range(D):
                s += Gam[mu][nu][lam] * hup[lam, nu] + Gam[nu][nu][lam] * hup[mu, lam]
        s -= sum(ginv[mu, nu] * sp.diff(htr, x[nu]) for nu in range(D))
        out.append(s)
    return out


def nabla_up_xi_up(g, ginv, Gam, xi, x):
    """(nabla^mu xi^nu) matrix."""
    D = len(x)
    cov = sp.zeros(D, D)  # nabla_alpha xi^nu
    for al in range(D):
        for nu in range(D):
            s = sp.diff(xi[nu], x[al])
            for lam in range(D):
                s += Gam[nu][al][lam] * xi[lam]
            cov[al, nu] = s
    return ginv * cov


def Q_grav_up(g, x, xi):
    """Q^{mu nu} without the -1/16pi factor: (nabla^mu xi^nu - nabla^nu xi^mu)."""
    ginv = inv_metric(g)
    Gam = christoffel(g, ginv, x)
    N = nabla_up_xi_up(g, ginv, Gam, xi, x)
    return N - N.T


def sqrt_minus_det(g):
    if g.is_diagonal():
        return sp.sqrt(-sp.prod([g[i, i] for i in range(g.shape[0])]))
    return sp.sqrt(-g.det())


def k_grav_tr(g, h, x, xi, i_t=0, i_r=1, eps=sp.Symbol('epsilon_var'), sqrtg=None):
    """
    sqrt(-g) k^{tr} (gravity part, times 16 pi, i.e. without the 1/16pi prefactor):
        16pi sqrt(-g) k^{tr} = delta(sqrt(-g) Qhat^{tr}) * (-1) + sqrt(-g) (Thetahat^r xi^t - Thetahat^t xi^r) * (-1)?
    Careful bookkeeping:
        Q^{mu nu} = -(1/16pi) Qhat^{mu nu},  Qhat = nabla^mu xi^nu - nabla^nu xi^mu
        Theta^mu  = +(1/16pi) Thetahat^mu
        k^{mu nu} sqrt(-g) = delta(sqrt(-g) Q^{mu nu}) - sqrt(-g)(Theta^mu xi^nu - Theta^nu xi^mu)
    Returns 16 pi * sqrt(-g) k^{tr}.
    """
    ge = g + eps * h
    Qhat_e = Q_grav_up(ge, x, xi)
    sg_e = sqrtg(ge, eps) if sqrtg is not None else sqrt_minus_det(ge)
    dQ = sp.diff(sg_e * Qhat_e[i_t, i_r], eps).subs(eps, 0)
    ginv = inv_metric(g)
    Gam = christoffel(g, ginv, x)
    Th = theta_grav(g, ginv, Gam, h, x)
    sg = sg_e.subs(eps, 0)
    term_theta = sg * (Th[i_t] * xi[i_r] - Th[i_r] * xi[i_t])
    return -dQ - term_theta


def sphere_metric(n, angles):
    """Round metric on S^n as a list of diagonal entries."""
    diag = []
    pref = 1
    for i in range(n):
        diag.append(pref)
        pref = pref * sp.sin(angles[i]) ** 2
    return diag


def sphere_volume(n):
    return 2 * sp.pi ** sp.Rational(n + 1, 2) / sp.gamma(sp.Rational(n + 1, 2))
