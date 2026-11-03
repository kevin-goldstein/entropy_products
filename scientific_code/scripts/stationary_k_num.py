r"""
stationary_k_num.py -- fast numerical evaluation of the Iyer-Wald 2-form k_xi in 4D for
    L = (1/16pi)[R - 2 g_ij dphi^i.dphi^j - f(phi) F.F] eps + (1/16pi) 2 ftilde(phi) F^F
on stationary-axisymmetric configurations in coordinates x = (t, r, th, ph); all fields depend on (r, th).

Only first derivatives of g, h, A, dA, phi, dphi are needed:
  16pi k_grav^{tr} sqrt(-g) = -delta( sqrt(-g) Qhat^{tr} ) - sqrt(-g) (Thhat^t xi^r - Thhat^r xi^t)
     Qhat^{mn} = g^{ma} g^{nb} (d_a xi_b - d_b xi_a),     xi_b = g_{bl} xi^l   (xi^l constant)
     Thhat^m   = g^{ma} g^{nb} ( d_n h_{ab} - d_a h_{nb} - Gamma^l_{nb} h_{al} + Gamma^l_{ab} h_{nl} )
  k_phi^{tr} sqrt(-g) = - sqrt(-g) (Th_phi^t xi^r - Th_phi^r xi^t),  Th_phi^m = -(1/4pi) g_ij dphi^i nabla^m phi^j
  k_A (th ph) = -(1/4pi)[ delta(xi.A G_{thph}) - xi^m (dA_m G_{thph} + dA_th G_{ph m} + dA_ph G_{m th}) ]
     G = f *F - ftilde F

The routine `make_k(...)` takes sympy expressions and returns a numerical function k(r, th) -> k_{th ph}.
"""
import itertools
import sympy as sp
import mpmath as mp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
X = [t, r, th, ph]


def _lam(expr, args):
    return sp.lambdify(args, expr, 'mpmath')


def make_k(g, Avec, scalars, gsig, fcoup, ftcoup, h, dAvec, dscalars, params):
    """Return python callable k(r, th, *param_values, xi) -> (k_grav, k_phi, k_A) for the (th,ph) component."""
    args = (r, th) + tuple(params)
    D = 4
    # component callables (value, d_r, d_th)
    def comp(M):
        return [[(_lam(M[i, j], args), _lam(sp.diff(M[i, j], r), args), _lam(sp.diff(M[i, j], th), args))
                 for j in range(D)] for i in range(D)]
    gC, hC = comp(g), comp(h)
    AC = [(_lam(Avec[i], args), _lam(sp.diff(Avec[i], r), args), _lam(sp.diff(Avec[i], th), args)) for i in range(D)]
    dAC = [(_lam(dAvec[i], args), _lam(sp.diff(dAvec[i], r), args), _lam(sp.diff(dAvec[i], th), args)) for i in range(D)]
    n = len(scalars)
    sC = [(_lam(scalars[i], args), _lam(sp.diff(scalars[i], r), args), _lam(sp.diff(scalars[i], th), args)) for i in range(n)]
    dsC = [_lam(dscalars[i], args) for i in range(n)]
    sym_s = sp.symbols('s0:%d' % max(n, 1))
    gsigC = _lam(gsig(list(sym_s[:n])), sym_s[:n]) if n else None
    fC = _lam(fcoup(list(sym_s[:n])), sym_s[:n])
    ftC = _lam(ftcoup(list(sym_s[:n])), sym_s[:n])
    dfC = [_lam(sp.diff(fcoup(list(sym_s[:n])), sym_s[i]), sym_s[:n]) for i in range(n)]
    dftC = [_lam(sp.diff(ftcoup(list(sym_s[:n])), sym_s[i]), sym_s[:n]) for i in range(n)]

    def evalM(C, rv, tv, pv, which):
        return mp.matrix([[C[i][j][which](rv, tv, *pv) for j in range(D)] for i in range(D)])

    def k(rv, tv, pv, xi):
        gm = evalM(gC, rv, tv, pv, 0)
        dg = [None, evalM(gC, rv, tv, pv, 1), evalM(gC, rv, tv, pv, 2), None]  # d_r, d_th
        hm = evalM(hC, rv, tv, pv, 0)
        dh = [None, evalM(hC, rv, tv, pv, 1), evalM(hC, rv, tv, pv, 2), None]
        gi = gm ** -1
        detg = mp.det(gm)
        sg = mp.sqrt(-detg)
        trh = sum(gi[a, b] * hm[a, b] for a in range(D) for b in range(D))
        dgi = -gi * hm * gi                     # delta g^{mn}
        dsg = sg * trh / 2                      # delta sqrt(-g)
        def d(Mlist, a, i, j):                  # partial_a of matrix component
            if a in (1, 2):
                return Mlist[a][i, j]
            return mp.mpf(0)
        # Christoffels Gamma^l_{ab}
        Gam = [[[mp.mpf(0)] * D for _ in range(D)] for _ in range(D)]
        for l in range(D):
            for a_ in range(D):
                for b in range(D):
                    s = 0
                    for m_ in range(D):
                        if gi[l, m_] != 0:
                            s += gi[l, m_] * (d(dg, a_, m_, b) + d(dg, b, m_, a_) - d(dg, m_, a_, b))
                    Gam[l][a_][b] = s / 2
        # gravity: Qhat^{tr} and its variation
        xil = [sum(gm[b, l] * xi[l] for l in range(D)) for b in range(D)]
        dxil = [sum(hm[b, l] * xi[l] for l in range(D)) for b in range(D)]
        def dxi_low(a_, b, gder):   # d_a xi_b with xi_b = g_bl xi^l
            return sum(d(gder, a_, b, l) * xi[l] for l in range(D))
        W = mp.matrix(D, D); dW = mp.matrix(D, D)
        for a_ in range(D):
            for b in range(D):
                W[a_, b] = dxi_low(a_, b, dg) - dxi_low(b, a_, dg)
                dW[a_, b] = dxi_low(a_, b, dh) - dxi_low(b, a_, dh)
        Qtr = sum(gi[0, a_] * gi[1, b] * W[a_, b] for a_ in range(D) for b in range(D))
        dQtr = sum((dgi[0, a_] * gi[1, b] + gi[0, a_] * dgi[1, b]) * W[a_, b] + gi[0, a_] * gi[1, b] * dW[a_, b]
                   for a_ in range(D) for b in range(D))
        dsgQ = dsg * Qtr + sg * dQtr
        # Theta^m (hat)
        Th = []
        for m_ in (0, 1):
            s = 0
            for a_ in range(D):
                if gi[m_, a_] == 0:
                    continue
                for n_ in range(D):
                    for b in range(D):
                        if gi[n_, b] == 0:
                            continue
                        term = d(dh, n_, a_, b) - d(dh, a_, n_, b)
                        term -= sum(Gam[l][n_][b] * hm[a_, l] for l in range(D))
                        term += sum(Gam[l][a_][b] * hm[n_, l] for l in range(D))
                        s += gi[m_, a_] * gi[n_, b] * term
            Th.append(s)
        kgrav = (-dsgQ - sg * (Th[0] * xi[1] - Th[1] * xi[0])) / (16 * mp.pi)
        # scalars
        svals = [sC[i][0](rv, tv, *pv) for i in range(n)]
        sder = [[mp.mpf(0), sC[i][1](rv, tv, *pv), sC[i][2](rv, tv, *pv), mp.mpf(0)] for i in range(n)]
        dsv = [dsC[i](rv, tv, *pv) for i in range(n)]
        kphi = mp.mpf(0)
        if n:
            G_ij = gsigC(*svals)
            Thp = []
            for m_ in (0, 1):
                s = 0
                for i in range(n):
                    for j in range(n):
                        if G_ij[i, j] != 0:
                            s += G_ij[i, j] * dsv[i] * sum(gi[m_, nu] * sder[j][nu] for nu in range(D))
                Thp.append(-s / (4 * mp.pi))
            kphi = -sg * (Thp[0] * xi[1] - Thp[1] * xi[0])
        # Maxwell
        Av = [AC[i][0](rv, tv, *pv) for i in range(D)]
        Ad = [[mp.mpf(0), AC[i][1](rv, tv, *pv), AC[i][2](rv, tv, *pv), mp.mpf(0)] for i in range(D)]
        dAv = [dAC[i][0](rv, tv, *pv) for i in range(D)]
        dAd = [[mp.mpf(0), dAC[i][1](rv, tv, *pv), dAC[i][2](rv, tv, *pv), mp.mpf(0)] for i in range(D)]
        F = mp.matrix(D, D); dF = mp.matrix(D, D)
        for m_ in range(D):
            for n_ in range(D):
                F[m_, n_] = Ad[n_][m_] - Ad[m_][n_]
                dF[m_, n_] = dAd[n_][m_] - dAd[m_][n_]
        Fup = gi * F * gi
        dFup = dgi * F * gi + gi * F * dgi + gi * dF * gi
        fv = fC(*svals); ftv = ftC(*svals)
        dfv = sum(dfC[i](*svals) * dsv[i] for i in range(n)) if n else 0
        dftv = sum(dftC[i](*svals) * dsv[i] for i in range(n)) if n else 0
        def star(Fu, s_):     # (*F)_{ab} = 1/2 eps_{mnab} F^{mn},  eps_{0123} = s_
            S = mp.matrix(D, D)
            for p in itertools.permutations(range(4)):
                sign = sp.combinatorics.Permutation(list(p)).signature()
                m_, n_, a_, b = p
                S[a_, b] += sign * s_ * Fu[m_, n_] / 2
            return S
        sF = star(Fup, sg)
        dsF = star(dFup, sg) + star(Fup, dsg)
        G = fv * sF - ftv * F
        dG = dfv * sF + fv * dsF - dftv * F - ftv * dF
        xiA = sum(xi[m_] * Av[m_] for m_ in range(D))
        dxiA = sum(xi[m_] * dAv[m_] for m_ in range(D))
        dQA = -(dxiA * G[2, 3] + xiA * dG[2, 3]) / (4 * mp.pi)
        xiTh = 0
        for m_ in range(D):
            if xi[m_] == 0:
                continue
            xiTh += xi[m_] * (dAv[m_] * G[2, 3] + dAv[2] * G[3, m_] + dAv[3] * G[m_, 2])
        xiTh = -xiTh / (4 * mp.pi)
        kA = dQA - xiTh
        return kgrav, kphi, kA
    return k


def K_of_r(kfun, rv, pv, xi, split=False):
    if split:
        return [2 * mp.pi * mp.quad(lambda tv, i=i: kfun(rv, tv, pv, xi)[i], [0, mp.pi / 2, mp.pi], method='gauss-legendre') for i in range(3)]
    return 2 * mp.pi * mp.quad(lambda tv: sum(kfun(rv, tv, pv, xi)), [0, mp.pi / 2, mp.pi], method='gauss-legendre')
