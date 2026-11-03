r"""
rotating_attractor_limit.py -- rotating averaged attractor equations, pointwise in theta, and their extremal limit.

A non-extremal stationary, axisymmetric, circular black hole is written (g_{r theta} = 0) as
    ds^2 = -N^2 dt^2 + g_rr dr^2 + g_thth dth^2 + g_phph (dph - w dt)^2 ,     A = At dt + Aph (dph - w dt).
For a function eps(theta) on the sphere we use the r-independent deformations
    (a) dg = 2 eps (-N^2 dt^2 + g_rr dr^2)       AdS2 block              <->  delta v1        (AGJST)
    (b) dg = 2 eps g_phph (dph - w dt)^2          fibre                   <->  delta v2
    (c) dg = 2 eps g_thth dth^2                   polar direction         <->  beta, constraint I(theta)
    (d) d phi^s = eps                             scalars                 <->  delta u_s
    (e) dA = eps (dph - w dt)                     gauge field             <->  delta b_i
They are smooth at both bifurcation surfaces.  (b), (c), (e) need eps = 0 at the poles (axis regularity, no Dirac
string), except the combination (b)+(c) with equal eps, which rescales the (theta, phi) block.  Prop. 5.6 gives
    I_X[eps] = 2 pi int_{r-}^{r+} dr int_0^pi dth  delta_X (sqrt(-g) L) = 0      for every admissible eps(theta),
i.e. the theta-Euler-Lagrange equations hold pointwise in theta once averaged over region II.

Checks (Kerr-Newman, and Kerr-Sen with its dilaton modulus on):
  (1) I_X = 0 while its gravity and matter parts are separately nonzero; the same integral over (r+, 3 r+) is not 0;
      profiles that violate the axis conditions, eps = 1 for (b), (c) or (e) alone, give I_X != 0 (controls);
  (2) as T -> 0, I_X / (kappa+ - kappa-) converges part by part to the first variation delta_X f of
      f = int dth dph sqrt(-g) L evaluated on the extremal near-horizon geometry (the function of AGJST,
      hep-th/0606244), which is therefore stationary: the AGJST equations are the T -> 0 limit;
  (3) the region-II Legendre identity I_II = -(T+S+ - T-S-) - (Omega+ - Omega-) J + (Phi- - Phi+) Q and its
      limit: alpha = lim (Omega+ - Omega-)/(kappa+ - kappa-), e = lim (Phi- - Phi+)/(kappa+ - kappa-),
      f = lim I_II/(kappa+ - kappa-) and S = 2 pi (e Q - alpha J - f) at T = 0; on the near-horizon geometry
      df/dalpha = -J and df/de = Q (AGJST's J is -J, their footnote 3).
Conventions as in the paper: L = (1/16 pi)[R - 2 g_ij dphi^i.dphi^j - f F.F - ft F.(*F)], G = 1.  The normalisation
kappa+ - kappa- = 2 pi (T+ - T-) (signed, kappa- < 0) divides out the normalisation of t.
"""
import checklib as ck
import numpy as np
import sympy as sp
from stationary_k_num import r, th

PI = np.pi
LAM = sp.Symbol('lam', positive=True)


# ----------------------------------------------------------------------------------------------- field evaluation
class Fields:
    """Symbolic stationary-axisymmetric fields in coordinates (t, r, th, ph); components depend on (r, th)."""

    def __init__(self, g, A, scalars, gsig, fcoup, ftcoup, params):
        self.g, self.A, self.s = g, list(A), list(scalars)
        self.gsig, self.fcoup, self.ftcoup, self.params = gsig, fcoup, ftcoup, tuple(params)
        self.args = (r, th) + self.params

    def deformation(self, kind, eps):
        """Return (h_ab, dA_a, ds_i) for the deformation of the given kind with profile eps(theta)."""
        g = self.g
        N2 = -(g[0, 0] * g[3, 3] - g[0, 3] ** 2) / g[3, 3]
        w = -g[0, 3] / g[3, 3]
        h = sp.zeros(4, 4)
        dA = [sp.Integer(0)] * 4
        ds = [sp.Integer(0)] * len(self.s)
        # near-horizon parameters (only meaningful on the near-horizon geometry, where r is rho):
        #   ds^2 = ... + g_phph (dph - alpha rho dt)^2,  A = e rho dt + b (dph - alpha rho dt)
        if kind == 'alpha':                      # delta alpha at fixed v1, v2, beta, e, b
            h[0, 3] = h[3, 0] = -r * g[3, 3]
            h[0, 0] = -2 * r * g[0, 3]
            dA[0] = -r * self.A[3]
            return h, dA, ds
        if kind == 'efield':                     # delta e at fixed everything else
            dA[0] = r
            return h, dA, ds
        if 'a' in kind:
            h[0, 0] += -2 * eps * N2
            h[1, 1] += 2 * eps * g[1, 1]
        if 'b' in kind:
            h[3, 3] += 2 * eps * g[3, 3]
            h[0, 3] += 2 * eps * g[0, 3]
            h[3, 0] += 2 * eps * g[0, 3]
            h[0, 0] += 2 * eps * g[0, 3] ** 2 / g[3, 3]
        if 'c' in kind:
            h[2, 2] += 2 * eps * g[2, 2]
        if 'e' in kind:
            dA[3] = eps
            dA[0] = -w * eps
        if kind.startswith('d'):                 # 'd0', 'd1', ...: shift of scalar number i
            ds[int(kind[1:])] = eps
        return h, dA, ds


def _lam(expr, args):
    f = sp.lambdify(args, expr, 'numpy')
    return f


def _jet(expr, args, order):
    """value and (r, th) derivatives up to the given order, as numpy callables."""
    out = {(0, 0): _lam(expr, args)}
    if order >= 1:
        out[(1, 0)] = _lam(sp.diff(expr, r), args)
        out[(0, 1)] = _lam(sp.diff(expr, th), args)
    if order >= 2:
        out[(2, 0)] = _lam(sp.diff(expr, r, 2), args)
        out[(1, 1)] = _lam(sp.diff(expr, r, th), args)
        out[(0, 2)] = _lam(sp.diff(expr, th, 2), args)
    return out


def _ev(f, R, TH, pv):
    v = f(R, TH, *pv)
    return np.broadcast_to(np.asarray(v, dtype=complex), R.shape).copy()


class Evaluator:
    """Numerical Lagrangian density (split into parts) for background + i*step*deformation (complex step)."""

    def __init__(self, F, kind=None, eps=None):
        self.F = F
        n = len(F.s)
        self.n = n
        args = F.args
        self.gj = [[_jet(F.g[a, b], args, 2) for b in range(4)] for a in range(4)]
        self.Aj = [_jet(F.A[a], args, 1) for a in range(4)]
        self.sj = [_jet(F.s[i], args, 1) for i in range(n)]
        ss = sp.symbols('s0:%d' % max(n, 1))[:n]
        if n:
            gsm = F.gsig(list(ss))
            self.gsE = [[_lam(gsm[i, j], ss) for j in range(n)] for i in range(n)]
        self.fC = _lam(F.fcoup(list(ss)), ss)
        self.ftC = _lam(F.ftcoup(list(ss)), ss)
        if kind is not None:
            h, dA, ds = F.deformation(kind, eps)
            self.hj = [[_jet(h[a, b], args, 2) for b in range(4)] for a in range(4)]
            self.dAj = [_jet(dA[a], args, 1) for a in range(4)]
            self.dsj = [_jet(ds[i], args, 1) for i in range(n)]
        else:
            self.hj = None

    def _metric(self, J, R, TH, pv):
        N = R.size
        G = np.zeros((N, 4, 4), complex)
        dG = np.zeros((N, 4, 4, 4), complex)
        ddG = np.zeros((N, 4, 4, 4, 4), complex)
        for a in range(4):
            for b in range(4):
                c = J[a][b]
                G[:, a, b] = _ev(c[(0, 0)], R, TH, pv)
                dG[:, 1, a, b] = _ev(c[(1, 0)], R, TH, pv)
                dG[:, 2, a, b] = _ev(c[(0, 1)], R, TH, pv)
                ddG[:, 1, 1, a, b] = _ev(c[(2, 0)], R, TH, pv)
                ddG[:, 1, 2, a, b] = ddG[:, 2, 1, a, b] = _ev(c[(1, 1)], R, TH, pv)
                ddG[:, 2, 2, a, b] = _ev(c[(0, 2)], R, TH, pv)
        return G, dG, ddG

    def _vec(self, J, R, TH, pv):
        N = R.size
        V = np.zeros((N, len(J)), complex)
        dV = np.zeros((N, 4, len(J)), complex)
        for i, c in enumerate(J):
            V[:, i] = _ev(c[(0, 0)], R, TH, pv)
            dV[:, 1, i] = _ev(c[(1, 0)], R, TH, pv)
            dV[:, 2, i] = _ev(c[(0, 1)], R, TH, pv)
        return V, dV

    def parts(self, R, TH, pv, step=0.0):
        """Lagrangian-density parts (per dt dr dth dph) at points (R, TH); with step != 0 the deformation is added
        as i*step*h and the derivative d/dlambda is returned (complex step)."""
        R = np.asarray(R, float).ravel()
        TH = np.asarray(TH, float).ravel()
        G, dG, ddG = self._metric(self.gj, R, TH, pv)
        A, dA = self._vec(self.Aj, R, TH, pv)
        if self.n:
            S, dS = self._vec(self.sj, R, TH, pv)
        if step:
            hG, hdG, hddG = self._metric(self.hj, R, TH, pv)
            G = G + 1j * step * hG
            dG = dG + 1j * step * hdG
            ddG = ddG + 1j * step * hddG
            hA, hdA = self._vec(self.dAj, R, TH, pv)
            A = A + 1j * step * hA
            dA = dA + 1j * step * hdA
            if self.n:
                hS, hdS = self._vec(self.dsj, R, TH, pv)
                S = S + 1j * step * hS
                dS = dS + 1j * step * hdS
        gi = np.linalg.inv(G)
        sg = np.sqrt(-np.linalg.det(G))
        # Christoffels and their derivatives
        GL = 0.5 * (dG.transpose(0, 2, 1, 3) + dG.transpose(0, 2, 3, 1) - dG)          # Gamma_{d,bc}
        Gam = np.einsum('nad,ndbc->nabc', gi, GL)
        dGL = 0.5 * (ddG.transpose(0, 1, 3, 2, 4) + ddG.transpose(0, 1, 3, 4, 2) - ddG)  # d_e Gamma_{d,bc}
        dgi = -np.einsum('nap,nepq,nqd->nead', gi, dG, gi)
        dGam = np.einsum('nead,ndbc->neabc', dgi, GL) + np.einsum('nad,nedbc->neabc', gi, dGL)
        Ric = (np.einsum('naabc->nbc', dGam) - np.einsum('ncaab->nbc', dGam)
               + np.einsum('naae,nebc->nbc', Gam, Gam) - np.einsum('nace,neab->nbc', Gam, Gam))
        Rs = np.einsum('nbc,nbc->n', gi, Ric)
        # Maxwell and axion terms
        Fl = dA.transpose(0, 1, 2) - dA.transpose(0, 2, 1)       # F_ab = d_a A_b - d_b A_a
        Fu = np.einsum('nac,nbd,ncd->nab', gi, gi, Fl)
        FF = np.einsum('nab,nab->n', Fl, Fu)
        FsF = (-4 / sg) * (Fl[:, 0, 1] * Fl[:, 2, 3] - Fl[:, 0, 2] * Fl[:, 1, 3] + Fl[:, 0, 3] * Fl[:, 1, 2])
        if self.n:
            sv = [S[:, i] for i in range(self.n)]
            fv = np.asarray(self.fC(*sv), complex) * np.ones_like(sg)
            ftv = np.asarray(self.ftC(*sv), complex) * np.ones_like(sg)
            gs = np.zeros((R.size, self.n, self.n), complex)
            for i in range(self.n):
                for j in range(self.n):
                    gs[:, i, j] = np.asarray(self.gsE[i][j](*sv), complex) * np.ones_like(sg)
            kin = np.einsum('nij,ncd,nci,ndj->n', gs, gi, dS, dS)
        else:
            fv = np.asarray(self.fC(), complex) * np.ones_like(sg)
            ftv = np.asarray(self.ftC(), complex) * np.ones_like(sg)
            kin = np.zeros_like(sg)
        # electric / magnetic split of F.F in the co-rotating frame (dt, dr, dth, dph - w dt):
        #   E_r = F_tr - w F_rph, E_th = F_tth - w F_thph, B_r = F_rph, B_th = F_thph
        w = -G[:, 0, 3] / G[:, 3, 3]
        N2 = -(G[:, 0, 0] * G[:, 3, 3] - G[:, 0, 3] ** 2) / G[:, 3, 3]
        Er = Fl[:, 0, 1] - w * Fl[:, 1, 3]
        Et = Fl[:, 0, 2] - w * Fl[:, 2, 3]
        FFe = -2 * (Er ** 2 / (N2 * G[:, 1, 1]) + Et ** 2 / (N2 * G[:, 2, 2]))
        FFm = 2 * (Fl[:, 1, 3] ** 2 / (G[:, 1, 1] * G[:, 3, 3]) + Fl[:, 2, 3] ** 2 / (G[:, 2, 2] * G[:, 3, 3]))
        self.split_error = ck.max_abs(FFe + FFm - FF) if FF.size else 0.0
        pre = sg / (16 * PI)
        out = dict(grav=pre * Rs, elec=-pre * fv * FFe, mag=-pre * fv * FFm, axion=-pre * ftv * FsF,
                   scalar=-2 * pre * kin)
        if step:
            out = {k: np.imag(v) / step for k, v in out.items()}
        else:
            out = {k: np.real(v) for k, v in out.items()}
        out['total'] = sum(out.values())
        return out


def integrate(ev, r0, r1, pv, step=1e-30, nr=32, nth=48):
    """2 pi int_{r0}^{r1} dr int_0^pi dth of each part (per unit t)."""
    xr, wr = np.polynomial.legendre.leggauss(nr)
    xt, wt = np.polynomial.legendre.leggauss(nth)
    rr = (r1 - r0) / 2 * xr + (r1 + r0) / 2
    tt = PI / 2 * xt + PI / 2
    RR, TT = np.meshgrid(rr, tt, indexing='ij')
    W = np.outer(wr * (r1 - r0) / 2, wt * PI / 2).ravel()
    P = ev.parts(RR, TT, pv, step=step)
    return {k: 2 * PI * np.sum(W * v) for k, v in P.items()}


def theta_integral(ev, rho, pv, step=1e-30, nth=64):
    """2 pi int_0^pi dth of each part at fixed r = rho (used on the near-horizon geometry)."""
    xt, wt = np.polynomial.legendre.leggauss(nth)
    tt = PI / 2 * xt + PI / 2
    P = ev.parts(np.full_like(tt, rho), tt, pv, step=step)
    return {k: 2 * PI * np.sum(wt * PI / 2 * v) for k, v in P.items()}


# ----------------------------------------------------------------------------------------------- near-horizon limit
def near_horizon(F, sub, r0, ct, OmH):
    """Extremal near-horizon limit: r = r0 + lam rho, t = ct tau / lam, ph = ph' + OmH t; returns Fields in
    coordinates (tau, rho, th, ph') (rho is the symbol r again).  sub: parameter values (exact) at extremality."""
    g = F.g.subs(sub)
    A = [x.subs(sub) for x in F.A]
    s = [x.subs(sub) for x in F.s]
    rho = r
    X = {r: r0 + LAM * rho}

    def lim(expr):
        e = sp.simplify(expr.subs(X))
        return sp.simplify(sp.limit(e, LAM, 0))

    gn = sp.zeros(4, 4)
    gn[0, 0] = lim((ct / LAM) ** 2 * (g[0, 0] + 2 * OmH * g[0, 3] + OmH ** 2 * g[3, 3]))
    gn[0, 3] = gn[3, 0] = lim((ct / LAM) * (g[0, 3] + OmH * g[3, 3]))
    gn[3, 3] = lim(g[3, 3])
    gn[1, 1] = lim(LAM ** 2 * g[1, 1])
    gn[2, 2] = lim(g[2, 2])
    At = A[0] + OmH * A[3]
    At0 = sp.simplify(At.subs(r, r0))
    An = [lim((ct / LAM) * (At - At0)), sp.Integer(0), sp.Integer(0), lim(A[3])]
    sn = [lim(x) for x in s]
    return Fields(gn, An, sn, F.gsig, F.fcoup, F.ftcoup, ()), sp.simplify(At0)


# ----------------------------------------------------------------------------------------------- models
def kerr_newman():
    from kerr_newman_check import g, Avec, M, a, Q
    F = Fields(g, Avec, [], lambda s: sp.zeros(0, 0), lambda s: sp.Integer(1), lambda s: sp.Integer(0), (M, a, Q))
    av, Qv = sp.Rational(3, 5), sp.Rational(4, 5)          # extremal at M = 1, r0 = 1

    def horizons(Mv):
        sq = np.sqrt(max(Mv ** 2 - float(av) ** 2 - float(Qv) ** 2, 0.0))   # 0 at extremality (no roundoff NaN)
        rp, rm = Mv + sq, Mv - sq
        hz = {}
        for lab, rh in (('+', rp), ('-', rm)):
            W = rh ** 2 + float(av) ** 2
            hz[lab] = dict(r=rh, S=PI * W, T=(rh - (rp + rm) / 2) / (2 * PI * W), Om=float(av) / W,
                           Phi=float(Qv) * rh / W)
        return hz, dict(J=Mv * float(av), Q=float(Qv))

    ext = dict(sub={M: 1, a: av, Q: Qv}, r0=sp.Integer(1), ct=1 + av ** 2, OmH=av / (1 + av ** 2), M_ext=1.0,
               pv=lambda Mv: (Mv, float(av), float(Qv)), S_ext=float(PI * (1 + av ** 2)))
    return 'Kerr-Newman (a = 3/5, Q = 4/5)', F, horizons, ext


def kerr_sen():
    from kerr_sen_check import g, Avec, scal, gsig, fcoup, ftcoup, M, a, Q, c
    F = Fields(g, Avec, scal, gsig, fcoup, ftcoup, (M, a, Q, c))
    av, cv = sp.Rational(1, 2), sp.Rational(3, 10)
    Qv = sp.exp(-cv)                                   # e^{2c} Q^2 = 1: extremal at M = 1 with r2 = 1, r0 = 1/2

    def horizons(Mv):
        r2 = 1.0 / Mv
        sq = np.sqrt(max((Mv - r2 / 2) ** 2 - float(av) ** 2, 0.0))
        rp, rm = Mv - r2 / 2 + sq, Mv - r2 / 2 - sq
        hz = {}
        for lab, rh in (('+', rp), ('-', rm)):
            hz[lab] = dict(r=rh, S=2 * PI * Mv * rh, T=(rh - (rp + rm) / 2) / (4 * PI * Mv * rh),
                           Om=float(av) / (2 * Mv * rh), Phi=float(sp.exp(2 * cv)) * float(Qv) / (2 * Mv))
        return hz, dict(J=Mv * float(av), Q=float(Qv))

    ext = dict(sub={M: 1, a: av, Q: Qv, c: cv}, r0=av, ct=sp.Integer(1), OmH=av, M_ext=1.0,
               pv=lambda Mv: (Mv, float(av), float(Qv), float(cv)), S_ext=float(2 * PI * av))
    return 'Kerr-Sen (a = 1/2, chi_inf = 3/10, e^{2 chi_inf} Q^2 = 1)', F, horizons, ext


# ----------------------------------------------------------------------------------------------- the checks
S2 = sp.sin(th) ** 2
TESTS = [('a', sp.Integer(1)), ('a', sp.cos(th) ** 2), ('b', S2), ('b', S2 * sp.cos(th) ** 2),
         ('c', S2), ('c', S2 * sp.cos(th) ** 2), ('bc', sp.Integer(1)), ('e', S2), ('e', S2 * sp.cos(th) ** 2)]
CONTROLS = [('b', sp.Integer(1)), ('c', sp.Integer(1)), ('e', sp.Integer(1))]


def run_model(model, extra_tests=(), deltas=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001)):
    name, F, horizons, ext = model
    tag = name.split(' (')[0]
    print('=' * 100)
    print(name)
    NH, At0 = near_horizon(F, ext['sub'], ext['r0'], ext['ct'], ext['OmH'])
    print('  near-horizon metric: g_tt =', sp.simplify(NH.g[0, 0]), ';  g_tph =', sp.simplify(NH.g[0, 3]))
    print('                       g_rr =', sp.simplify(NH.g[1, 1]), ';  g_thth =', sp.simplify(NH.g[2, 2]))
    print('  near-horizon gauge field: A_t =', sp.simplify(NH.A[0]), ';  A_ph =', sp.simplify(NH.A[3]))
    alpha_nh = float(sp.simplify(-NH.g[0, 3] / NH.g[3, 3] / r).subs(th, 0.7))
    e_nh = float(sp.simplify((NH.A[0] + (-NH.g[0, 3] / NH.g[3, 3]) * NH.A[3]) / r).subs(th, 0.7))
    print('  alpha (from g_tph = -alpha rho g_phph) = %.12f ;  e (from A = e rho dt + b (dph - alpha rho dt)) = %.12f'
          % (alpha_nh, e_nh))
    tests = TESTS + list(extra_tests)
    # (1) finite temperature
    Mv = ext['M_ext'] * (1 + 0.1)
    hz, ch = horizons(Mv)
    rp, rm = hz['+']['r'], hz['-']['r']
    print('\n  (1) finite T: M = %.3f, r+ = %.6f, r- = %.6f.  Region-II integrals of delta_X(sqrt(-g) L):' % (Mv, rp, rm))
    for kind, eps in tests:
        ev = Evaluator(F, kind, eps)
        I2 = integrate(ev, rm, rp, ext['pv'](Mv))
        Iout = integrate(ev, rp, 3 * rp, ext['pv'](Mv))
        parts = '  '.join('%s=%+.6f' % (k, v) for k, v in I2.items() if k != 'total' and abs(v) > 1e-14)
        print('     %-3s eps=%-22s total = %+.2e   [%s]   outside (r+,3r+): %+.5f' % (
            kind, str(eps), I2['total'], parts, Iout['total']))
        ck.small('%s (1) %s eps=%s: region-II integral vanishes' % (tag, kind, eps), I2['total'], 1e-11)
        ck.large('%s (1) %s eps=%s: the same integral over (r+, 3r+) does not' % (tag, kind, eps), Iout['total'], 1e-3)
    print('     controls that violate the axis conditions (eps = 1 for (b), (c), (e) alone; (b)+(c) is allowed):')
    for kind, eps in CONTROLS:
        I2 = integrate(Evaluator(F, kind, eps), rm, rp, ext['pv'](Mv))
        print('     %-3s eps=%-22s total = %+.6f' % (kind, str(eps), I2['total']))
        ck.large('%s negative control: %s eps=%s violates the axis conditions and the region-II integral is nonzero'
                 % (tag, kind, eps), I2['total'], 1e-2)
    # (2) limit T -> 0, compared with the first variation of f on the near-horizon geometry
    print('\n  (2) T -> 0: I_X/(kappa+ - kappa-) part by part, against delta_X f on the extremal near-horizon geometry')
    for kind, eps in tests:
        evnh = Evaluator(NH, kind, eps)
        fx1 = theta_integral(evnh, 1.0, ())
        fx2 = theta_integral(evnh, 2.0, ())
        keys = [k for k in ('grav', 'elec', 'mag', 'scalar', 'axion') if abs(fx1[k]) > 1e-12]
        ev = Evaluator(F, kind, eps)
        rows = []
        for dlt in deltas:
            Mv = ext['M_ext'] * (1 + dlt)
            hz, ch = horizons(Mv)
            I2 = integrate(ev, hz['-']['r'], hz['+']['r'], ext['pv'](Mv), nr=24)
            norm = 2 * PI * (hz['+']['T'] - hz['-']['T'])
            rows.append((dlt, {k: I2[k] / norm for k in keys}, I2['total'] / norm))
        print('     %-3s eps=%-22s NH (rho=1): %s   total %+.1e   [rho-independent: max diff %.1e]' % (
            kind, str(eps), '  '.join('%s=%+.8f' % (k, fx1[k]) for k in keys), fx1['total'],
            ck.max_abs(fx1[k] - fx2[k] for k in fx1)))
        for dlt, vals, tot in rows:
            print('         delta=%-6g  %s   total %+.1e' % (dlt, '  '.join('%s=%+.8f' % (k, vals[k]) for k in keys), tot))
        ck.small('%s (2) %s eps=%s: delta_X f = 0 on the near-horizon geometry' % (tag, kind, eps), fx1['total'], 1e-13)
        ck.small('%s (2) %s eps=%s: near-horizon parts independent of rho' % (tag, kind, eps),
                 ck.max_abs(fx1[k] - fx2[k] for k in fx1), 1e-13)
        ck.small('%s (2) %s eps=%s: normalised region-II total at every delta (24-point radial quadrature)' % (tag, kind, eps),
                 ck.max_abs(tot for _, _, tot in rows), 1e-7)
        devs = [ck.max_abs(vals[k] - fx1[k] for k in keys) for _, vals, _ in rows]
        ck.true('%s (2) %s eps=%s: each part of I_X/(kappa+ - kappa-) converges to delta_X f as delta -> 0'
                % (tag, kind, eps), all(devs[i + 1] < devs[i] for i in range(len(devs) - 1)) and devs[-1] < 5e-3,
                'largest deviation %.1e at delta = %g' % (devs[-1], deltas[-1]))
    # (3) Legendre identity at finite T and its limit
    print('\n  (3) region-II Legendre identity  I_II = -(T+S+ - T-S-) - (Om+ - Om-) J + (Phi- - Phi+) Q  and its limit'
          ' (norm = kappa+ - kappa- = 2 pi (T+ - T-))')
    ev0 = Evaluator(F)
    lim = []
    for dlt in deltas:
        Mv = ext['M_ext'] * (1 + dlt)
        hz, ch = horizons(Mv)
        I2 = integrate(ev0, hz['-']['r'], hz['+']['r'], ext['pv'](Mv), step=0.0)['total']
        p, m = hz['+'], hz['-']
        rhs = -(p['T'] * p['S'] - m['T'] * m['S']) - (p['Om'] - m['Om']) * ch['J'] + (m['Phi'] - p['Phi']) * ch['Q']
        norm = 2 * PI * (p['T'] - m['T'])
        print('     delta=%-6g  I_II = %+.12f   horizon side = %+.12f   |   I_II/norm = %+.8f   (Om+-Om-)/norm = %+.8f'
              '   (Phi--Phi+)/norm = %+.8f' % (dlt, I2, rhs, I2 / norm, (p['Om'] - m['Om']) / norm,
                                               (m['Phi'] - p['Phi']) / norm))
        ck.close('%s (3) delta = %g: region-II Legendre identity' % (tag, dlt), I2, rhs, 1e-9)
        lim.append((I2 / norm, (p['Om'] - m['Om']) / norm, (m['Phi'] - p['Phi']) / norm))
    fnh = theta_integral(Evaluator(NH), 1.0, (), step=0.0)['total']
    hz, ch = horizons(ext['M_ext'])
    print('     near horizon: f = %+.8f, alpha = %+.8f, e = %+.8f;  S_ext = %.8f;  2 pi (e Q - alpha J - f) = %.8f'
          % (fnh, alpha_nh, e_nh, ext['S_ext'], 2 * PI * (e_nh * ch['Q'] - alpha_nh * ch['J'] - fnh)))
    dfa = theta_integral(Evaluator(NH, 'alpha', None), 1.0, ())['total']
    dfe = theta_integral(Evaluator(NH, 'efield', None), 1.0, ())['total']
    print('     charges from f (AGJST 2.12): df/dalpha = %+.10f  (J = %.10f);   df/de = %+.10f  (Q = %.10f)'
          % (dfa, ch['J'], dfe, ch['Q']))
    for j, (lab, target) in enumerate((('I_II/norm -> f', fnh), ('(Om+ - Om-)/norm -> alpha', alpha_nh),
                                       ('(Phi- - Phi+)/norm -> e', e_nh))):
        dv = [abs(row[j] - target) for row in lim]
        ok = dv[-1] < 5e-3 and (all(dv[i + 1] < dv[i] for i in range(len(dv) - 1)) or ck.max_abs(dv) < 1e-9)
        ck.true('%s (3) limit %s' % (tag, lab), ok, 'deviation %.1e at delta = %g' % (dv[-1], deltas[-1]))
    ck.close('%s (3) S_ext = 2 pi (e Q - alpha J - f)' % tag, 2 * PI * (e_nh * ch['Q'] - alpha_nh * ch['J'] - fnh), ext['S_ext'], 1e-8)
    ck.close('%s (3) AGJST (2.12): df/dalpha = -J' % tag, dfa, -ch['J'], 1e-8)
    ck.close('%s (3) AGJST (2.12): df/de = Q' % tag, dfe, ch['Q'], 1e-8)


if __name__ == '__main__':
    run_model(kerr_newman())
    run_model(kerr_sen(), extra_tests=[('d0', sp.Integer(1)), ('d0', sp.cos(th) ** 2), ('d1', sp.cos(th)),
                                       ('d1', sp.cos(th) ** 3)])
    ck.finish()
